import torch
import re
import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------
CHECKPOINT_PATH = "./math_verifier_model/checkpoint-12500"
MAX_SEQ_LEN = 2048
VAL_SIZE = 1000  # number of validation samples to evaluate

# -----------------------------
# Load model and tokenizer
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
model.eval().to("cuda")

print(f"Loaded model from {CHECKPOINT_PATH}")

# -----------------------------
# Load and preprocess validation data
# -----------------------------
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

def clean_solution(example):
    solution = re.sub(r"```.*?```", "", example['solution'], flags=re.DOTALL)
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    example['solution'] = solution
    example['is_correct'] = str(example['is_correct'])
    return example

shuffled_dataset = dataset.shuffle(seed=42)
validation_dataset = shuffled_dataset.select(range(100000, 101000))

# -----------------------------
# Prompt and parsing functions
# -----------------------------
inference_prompt = (
    "Decide if the given answer is correct. Respond with True or False only.\n"
    "Question: {}\n"
    "Proposed Answer: {}\n"
    "Solution: {}\n"
    "Correctness: "
)

# def parse_output(text):
#     text = text.lower()
#     if "true" in text: return True
#     if "false" in text: return False
    # return False

def parse_output(response_text, answer):
    # Find the text after "Output:"
    output_part = response_text.split("Correctness:")[-1]
    # Check if "True" is in that part, case-insensitively
    if 'true' in output_part.lower():
        return True
    elif 'false' in output_part.lower():
        return False
    elif answer in output_part.lower():
        return True
    ans = output_part.split('<|end_of_text|>')[0].strip()
    try:
        if int(float(ans)) == 1:
            return True
    except:
        pass
    return False

# -----------------------------
# Evaluation
# -----------------------------
correct = 0
for ex in tqdm.tqdm(validation_dataset):
    prompt = inference_prompt.format(ex["question"], ex["answer"], ex["solution"])
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=6)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = parse_output(decoded, answer=ex["answer"])
    true = (ex["is_correct"] == True)
    correct += int(pred == true)

accuracy = correct / len(validation_dataset)
print(f"\nValidation Accuracy ({VAL_SIZE} samples): {accuracy:.4f}")
