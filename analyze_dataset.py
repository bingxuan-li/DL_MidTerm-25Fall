import re
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
train_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# -----------------------------
# Cleaning function
# -----------------------------
def clean_solution(solution: str):
    solution = re.sub(r"```.*?```", "", solution, flags=re.DOTALL)
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    return solution

# -----------------------------
# Compute label distribution
# -----------------------------
labels = [ex["is_correct"] for ex in train_dataset]
label_counts = Counter(labels)
total = sum(label_counts.values())
print("\nLabel distribution:")
for k, v in label_counts.items():
    print(f"  {k}: {v} ({v/total*100:.2f}%)")

# -----------------------------
# Tokenizer for length analysis
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3-8B-bnb-4bit")

def avg_length(field):
    lens = [len(tokenizer(ex[field])["input_ids"]) for ex in train_dataset.select(range(10000))]
    return np.mean(lens)

print("\nAverage token lengths (first 10000 samples):")
print(f"  Questions: {avg_length('question'):.1f}")
print(f"  Answers:   {avg_length('answer'):.1f}")
print(f"  Solutions: {avg_length('solution'):.1f}")

# -----------------------------
# Question type breakdown (simple heuristic)
# -----------------------------
keywords = {
    "algebra": ["equation", "variable", "polynomial"],
    "geometry": ["triangle", "circle", "area", "angle"],
    "calculus": ["derivative", "integral", "limit"],
    "probability": ["probability", "expected", "distribution"],
    "arithmetic": ["sum", "product", "number", "integer"]
}

def detect_type(q):
    q_lower = q.lower()
    for k, kws in keywords.items():
        if any(kw in q_lower for kw in kws):
            return k
    return "other"

types = [detect_type(ex["question"]) for ex in train_dataset.select(range(5000))]
type_counts = Counter(types)

print("\nQuestion type breakdown (first 5k samples):")
for k, v in type_counts.items():
    print(f"  {k}: {v/len(types)*100:.1f}%")

# -----------------------------
# Example output to paste into LaTeX
# -----------------------------
print("\nSuggested LaTeX snippet:\n")
print(f"Dataset size: {len(train_dataset)} training, {len(test_dataset)} test samples.")
for k, v in label_counts.items():
    print(f"{k}: {v/total*100:.2f}%")
