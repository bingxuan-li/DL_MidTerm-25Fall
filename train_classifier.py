# train_rank_verifier.py
import os
import re
import math
import random
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from unsloth import FastLanguageModel  # your loader
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "unsloth/Llama-3-8B-bnb-4bit"
OUTPUT_DIR = "./math_rank_verifier"
MAX_SEQ_LEN = 1024          # Use 1024 or 2048 depending on memory
BATCH_SIZE = 1              # batch here = number of pairs per step
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
EPOCHS = 1
MARGIN = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_POS_PER_Q = 3           # limit pos per question to avoid combinatorial explosion
MAX_NEG_PER_Q = 3
MAX_PAIRS = 100000         # max number of pairs to build (cap)

wandb.init(project="math-verifier-ranking", name="llama3-ranking")

# -----------------------------
# Load model + tokenizer (4-bit + LoRA)
# -----------------------------
print("Loading model...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
# The returned object is the Unsloth wrapper; we'll attempt to access underlying HF model when needed.

# -----------------------------
# Build pair dataset
# -----------------------------
raw = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
print("Loaded raw dataset:", raw)

# Simple cleaning
def clean_solution(text):
    s = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    s = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", s)
    return s

# Group examples by question, collect pos/neg lists
groups = {}
for ex in tqdm(raw, desc="Grouping dataset"):
    q = ex["question"]
    groups.setdefault(q, {"pos": [], "neg": []})
    sol = clean_solution(ex["solution"] or "")
    ans = ex.get("answer", "")
    is_corr = bool(ex["is_correct"]) if "is_correct" in ex else (str(ex.get("label","False")).lower()=="true")
    entry = {"answer": ans, "solution": sol}
    if is_corr:
        groups[q]["pos"].append(entry)
    else:
        groups[q]["neg"].append(entry)

# Build pairs (sample to limit explosion)
pairs = []
for q, g in groups.items():
    if not g["pos"] or not g["neg"]:
        continue
    pos_list = g["pos"][:MAX_POS_PER_Q]
    neg_list = g["neg"][:MAX_NEG_PER_Q]
    for p in pos_list:
        for n in neg_list:
            pairs.append({
                "question": q,
                "pos_answer": p["answer"],
                "pos_solution": p["solution"],
                "neg_answer": n["answer"],
                "neg_solution": n["solution"],
            })
    if len(pairs) >= MAX_PAIRS:
        break

print(f"Built {len(pairs)} pairs (capped at {MAX_PAIRS})")

random.shuffle(pairs)
# split train / val
split_idx = int(len(pairs) * 0.99)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

# -----------------------------
# PyTorch Dataset + Collator
# -----------------------------
class PairDataset(Dataset):
    def __init__(self, pair_list):
        self.pairs = pair_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def build_prompt(question, answer, solution):
    return (
        "Evaluate the correctness of the proposed answer. Output a score: higher => more correct.\n"
        f"Question: {question}\n"
        f"Proposed Answer: {answer}\n"
        f"Solution: {solution}\n"
        "Score:"
    )

class PairCollator:
    def __init__(self, tokenizer, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pos_prompts = [build_prompt(x["question"], x["pos_answer"], x["pos_solution"]) for x in batch]
        neg_prompts = [build_prompt(x["question"], x["neg_answer"], x["neg_solution"]) for x in batch]
        pos_tok = self.tokenizer(pos_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        neg_tok = self.tokenizer(neg_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        return {
            "pos_input_ids": pos_tok["input_ids"],
            "pos_attention_mask": pos_tok["attention_mask"],
            "neg_input_ids": neg_tok["input_ids"],
            "neg_attention_mask": neg_tok["attention_mask"],
        }

train_dataset = PairDataset(train_pairs)
val_dataset = PairDataset(val_pairs)
collator = PairCollator(tokenizer, max_length=MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# -----------------------------
# Ranking model wrapper
# -----------------------------
import torch.nn as nn

class RankVerifier(nn.Module):
    def __init__(self, base_wrapper, hidden_pool="last"):
        super().__init__()
        # base_wrapper is your FastLanguageModel wrapper. Try to find underlying HF model.
        # We will call base_wrapper.model if exists, otherwise base_wrapper
        if hasattr(base_wrapper, "model"):
            self.base = base_wrapper.model
        else:
            self.base = base_wrapper
        self.tokenizer = tokenizer

        # enable checkpointing
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()
        
        orig_hidden = self.base.config.hidden_size
        proj_hidden = int(orig_hidden * 1.0)

        # self.proj = nn.Linear(orig_hidden, proj_hidden, bias=False)
        self.score_head = nn.Linear(proj_hidden, 1)

    def encode_last_hidden(self, input_ids, attention_mask):
        # We call the base model's forward and extract last hidden state
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        # Pool: take last non-masked token embedding per sample
        # find lengths
        lengths = attention_mask.sum(dim=1) - 1  # index of last token
        batch_size = input_ids.size(0)
        device = input_ids.device
        idx = lengths.view(batch_size, 1, 1).expand(batch_size, 1, last_hidden.size(-1))
        last_tok_emb = last_hidden.gather(1, idx).squeeze(1)  # (B, H)
        return last_tok_emb

    def forward(self, input_ids, attention_mask):
        # return (B, 1) score tensor
        with torch.no_grad():  # base is 4-bit and may be frozen; still need gradients for score head only if LoRA present
            # If LoRA is applied, base params may be trainable; we will not disable grads here.
            # However some wrappers require different handling; keep this flexible.
            pass
        emb = self.encode_last_hidden(input_ids, attention_mask)
        emb = emb.float()   # convert to fp32
        # emb = self.proj(emb)           # shrink first
        score = self.score_head(emb)  # (B, 1)
        return score

# instantiate
rank_model = RankVerifier(base_model).to(DEVICE)
# set trainability:
# - if LoRA adapters are trainable on base_model, they are in base_model parameters
# - we want score_head params trainable; choose whether to also fine-tune LoRA
for p in rank_model.score_head.parameters():
    p.requires_grad = True

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
params = [p for p in rank_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
warmup_steps = max(1, int(0.00 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# -----------------------------
# Ranking loss (hinge) and train loop
# -----------------------------
def hinge_margin_loss(pos_score, neg_score, margin=MARGIN):
    # pos_score, neg_score: (B, 1)
    diff = pos_score - neg_score
    loss = torch.clamp(margin - diff, min=0.0)
    return loss.mean()

def validate(rank_model, val_loader, device, n_batches=50):
    rank_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            pos_scores = rank_model(pos_ids, pos_mask)
            neg_scores = rank_model(neg_ids, neg_mask)
            correct += (pos_scores > neg_scores).sum().item()
            total += pos_scores.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc

# Training
print("Starting training...")
global_step = 0
rank_model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        pos_ids = batch["pos_input_ids"].to(DEVICE)
        pos_mask = batch["pos_attention_mask"].to(DEVICE)
        neg_ids = batch["neg_input_ids"].to(DEVICE)
        neg_mask = batch["neg_attention_mask"].to(DEVICE)

        pos_scores = rank_model(pos_ids, pos_mask)
        neg_scores = rank_model(neg_ids, neg_mask)

        loss = hinge_margin_loss(pos_scores, neg_scores, margin=MARGIN)
        loss = loss / GRAD_ACCUM
        loss.backward()

        running_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % 10 == 0:
                wandb.log({"train_loss": running_loss / 10, "step": global_step})
                print(f"Step {global_step:6d} | avg loss {running_loss/10:.6f}")
                running_loss = 0.0
                wandb.log({"learning_rate": scheduler.get_last_lr()[0], "step": global_step})

            # run validation every N steps
            if global_step % 500 == 0:
                val_acc = validate(rank_model, val_loader, DEVICE, n_batches=50)
                wandb.log({"val_pair_accuracy": val_acc, "global_step": global_step})
                print(f">>> Validation pairwise accuracy (partial): {val_acc:.4f}")
                
                rank_model.train()
            global_step += 1

    # end epoch validation
    val_acc = validate(rank_model, val_loader, DEVICE, n_batches=200)
    wandb.log({"val_pair_accuracy_epoch": val_acc, "epoch": epoch})
    print(f"End epoch {epoch}: val pairwise accuracy (partial): {val_acc:.4f}")

# -----------------------------
# Save score head
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(rank_model.score_head.state_dict(), os.path.join(OUTPUT_DIR, "score_head.pt"))
print(f"Saved score_head to {OUTPUT_DIR}/score_head.pt")

wandb.finish()
print("Training complete. Model artifacts saved to", OUTPUT_DIR)

rank_model.eval().cuda()

# -----------------------------
# Infer on test set
# -----------------------------
test_set = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")

preds = []
with torch.no_grad():
    for ex in tqdm(test_set, desc="Scoring test examples"):
        q = ex["question"]
        ans = ex.get("answer", "")              # dataset includes answer
        sol = clean_solution(ex["solution"])

        prompt = build_prompt(q, ans, sol)
        tok = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to("cuda")

        score = rank_model(tok["input_ids"], tok["attention_mask"]).item()
        preds.append(score > 0)  # threshold at 0

# -----------------------------
# Save submission
# -----------------------------
submission = pd.DataFrame({
    "ID": range(len(preds)),
    "is_correct": preds
})
submission.to_csv("submission.csv", index=False)

print("submission.csv saved. Upload to Kaggle!")
