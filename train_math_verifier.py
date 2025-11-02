import os
import re
import torch
import pandas as pd
import wandb, tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "unsloth/Llama-3-8B-bnb-4bit"
OUTPUT_DIR = "./math_verifier_model"
MAX_SEQ_LEN = 2048  # longer sequences to cover full solution
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
EPOCHS = 1  # increase if time/memory allows

wandb.init(
    project="math-verifier",
    name="llama3-8b-qlora-optimized",
    config={
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
    },
)

# -----------------------------
# Load model + LoRA
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
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
# Load dataset
# -----------------------------
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train[:100]")

# -----------------------------
# Data cleaning + preprocessing
# -----------------------------
def clean_solution(example):
    # Remove code fences
    solution = re.sub(r"```.*?```", "", example['solution'], flags=re.DOTALL)
    # Normalize numbers to 4 decimal places
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    example['solution'] = solution
    # Convert boolean label to string for SFT
    example['is_correct'] = str(example['is_correct'])
    return example

dataset = dataset.map(clean_solution)

# -----------------------------
# Prompt formatting
# -----------------------------
EOS_TOKEN = tokenizer.eos_token  # usually ''

training_prompt = """Decide if the given answer is correct. Respond with True or False only. \nQuestion: {}\nProposed Answer: {}\nSolution:{}\nCorrectness:{}"""

def format_prompt_with_eos(example):
    text = training_prompt.format(
        example['question'],
        example['answer'],
        example['solution'],
        str(example['is_correct'])
    ) + EOS_TOKEN
    return {"text": text}

dataset = dataset.map(format_prompt_with_eos)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    num_train_epochs=EPOCHS,
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    report_to="wandb",
    remove_unused_columns=False,
)

# -----------------------------
# SFT Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
wandb.finish()
print(f"Model saved to {OUTPUT_DIR}")


# Load the official test set
test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")


# Instructional prompt template for test
test_prompt_template = """Decide if the given answer is correct. Respond with True or False only. \nQuestion: {}\nProposed Answer: {}\nSolution:{}\nCorrectness:"""

# A simple function to parse 'True' or 'False' from the model's raw output
def parse_output(response_text):
    output_part = response_text.split("Correctness:")[-1]
    if 'true' in output_part.lower():
        return True
    elif 'false' in output_part.lower():
        return False
    else:
        return True  # Default to True if unclear

predictions = []

# Loop through the test dataset and generate a prediction for each example
for example in tqdm.tqdm(test_dataset):
    question = example["question"]
    answer = example["answer"]
    solution = example["solution"]

    # Format the prompt with EOS
    prompt = test_prompt_template.format(question, answer, solution)

    # Tokenize and move to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate output from the model (max 8 tokens to get True/False)
    outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Parse True/False prediction
    prediction = parse_output(response_text)
    predictions.append(prediction)

# Create the submission DataFrame
submission = pd.DataFrame({
    "ID": range(len(test_dataset)),
    "is_correct": predictions
})

# Save the CSV
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")
