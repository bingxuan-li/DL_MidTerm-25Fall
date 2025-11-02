import os
import wandb
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")

def format_test_prompt(example):
    return {
        "text": (
            "Decide if the given answer is correct. Respond with True or False only.\n\n"
            f"Question: {example['question']}\n"
            f"Proposed Answer: {example['answer']}\n"
            f"Solution: {example['solution']}\n"
        )
    }

test_dataset = test_dataset.map(format_test_prompt)

MODEL_DIR = "./math_verifier_model"

print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)
preds = trainer.predict(test_dataset)
# Convert logits to True/False
pred_labels = [pred > 0.5 for pred in preds.predictions]

import pandas as pd
submission = pd.DataFrame({
    "ID": range(len(test_dataset)),
    "is_correct": pred_labels
})
submission.to_csv("submission.csv", index=False)

