import os
import re
import torch
import pandas as pd
import wandb, tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "unsloth/Llama-3-8B-bnb-4bit"
OUTPUT_DIR = "./math_verifier_model"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
EPOCHS = 1

wandb.init(
    project="math-verifier",
    name="llama3-8b-qlora-optimized",
)

# -----------------------------
# Load model + LoRA
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing=True,
)

# -----------------------------
# Dataset split
# -----------------------------
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
shuffled_dataset = full_dataset.shuffle(seed=42)
train_dataset = shuffled_dataset.select(range(100000))
validation_dataset = shuffled_dataset.select(range(100000, 101000))

# -----------------------------
# Cleaning
# -----------------------------
def clean_solution(example):
    solution = re.sub(r"```.*?```", "", example['solution'], flags=re.DOTALL)
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    example['solution'] = solution
    example['is_correct'] = str(example['is_correct'])
    return example

train_dataset = train_dataset.map(clean_solution)
validation_dataset = validation_dataset.map(clean_solution)

# -----------------------------
# Prompt formatting
# -----------------------------
EOS = tokenizer.eos_token

training_prompt = (
    "Decide if the given answer is correct. Respond with True or False only.\n"
    "Question: {}\n"
    "Proposed Answer: {}\n"
    "Solution: {}\n"
    "Correctness: {}"
)

def format_prompt(example):
    return {
        "text": training_prompt.format(
            example["question"],
            example["answer"],
            example["solution"],
            example["is_correct"]
        ) + EOS
    }

train_dataset = train_dataset.map(format_prompt)
validation_dataset = validation_dataset.map(format_prompt)

# -----------------------------
# Custom validation metric
# -----------------------------
### ADDED: parsing fn
def parse_output(text):
    text = text.lower()
    if "true" in text: return True
    if "false" in text: return False
    return False  

### ADDED: validation callback
class ValCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        correct = 0
        total = len(validation_dataset)

        for ex in validation_dataset.select(range(500)):  # first 500 only
            prompt = ex["text"].rsplit("Correctness:", 1)[0] + "Correctness:"
            inp = tokenizer([prompt], return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=4)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = parse_output(text)
            true = (ex["is_correct"].lower() == "true")
            correct += int(pred == true)

        val_acc = correct / 500
        wandb.log({"val_accuracy": val_acc})
        print(f"\nValidation Accuracy (500 samples): {val_acc:.4f}\n")

# -----------------------------
# Training args
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    num_train_epochs=EPOCHS,
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    report_to="wandb",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

trainer.add_callback(ValCallback())  ### add callback

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
wandb.finish()
print(f"Model saved to {OUTPUT_DIR}")
