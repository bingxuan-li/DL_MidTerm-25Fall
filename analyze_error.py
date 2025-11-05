import torch, re, tqdm, pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel

# -----------------------------
# Config
# -----------------------------
save_path = "./math_verifier_model/checkpoint-12500"
MAX_SEQ_LEN = 2048
N_EXAMPLES = 10  # how many misclassified examples to print

# -----------------------------
# Load model + tokenizer
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=save_path,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
model.eval().to("cuda")
print(f"Model and tokenizer loaded from: {save_path}")

# -----------------------------
# Load validation or test split with labels
# -----------------------------
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")

def clean_solution(example):
    solution = re.sub(r"```.*?```", "", example["solution"], flags=re.DOTALL)
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    example["solution"] = solution
    example["is_correct"] = str(example["is_correct"])
    return example

dataset = dataset.map(clean_solution)

# -----------------------------
# Prompt template and parser
# -----------------------------
inference_prompt = (
    "Decide if the given answer is correct. Respond with True or False only.\n"
    "Question: {}\n"
    "Proposed Answer: {}\n"
    "Solution: {}\n"
    "Correctness: "
)

def parse_output(response_text, answer):
    output_part = response_text.split("Correctness:")[-1]
    if "true" in output_part.lower():
        return True
    elif "false" in output_part.lower():
        return False
    elif answer.lower() in output_part.lower():
        return True
    ans = output_part.split("<|end_of_text|>")[0].strip()
    try:
        if int(float(ans)) == 1:
            return True
    except:
        pass
    return False

# -----------------------------
# Run inference and collect errors
# -----------------------------
errors = []
for example in tqdm.tqdm(dataset.select(range(100)), desc="Evaluating"):
    question = example["question"]
    answer = example["answer"]
    solution = example["solution"]
    true_label = example["is_correct"] == True

    prompt = inference_prompt.format(question, answer, solution)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0]
    pred = parse_output(response_text, answer)

    if pred != true_label:
        errors.append({
            "question": question,
            "answer": answer,
            "solution": solution[:300] + ("..." if len(solution) > 300 else ""),
            "true_label": true_label,
            "predicted": pred,
            "raw_output": response_text,
        })

# -----------------------------
# Show sample misclassifications
# -----------------------------
print(f"\nTotal misclassifications: {len(errors)} / {len(dataset)}\n")
for e in errors[:N_EXAMPLES]:
    print("==== Misclassified Example ====")
    print("Question:", e["question"])
    print("Answer:", e["answer"])
    print("True Label:", e["true_label"], " | Predicted:", e["predicted"])
    print("Model Output:", e["raw_output"].split("Correctness:")[-1].strip())
    print()

# -----------------------------
# Optional: save to CSV for report
# -----------------------------
pd.DataFrame(errors).to_csv("error_analysis.csv", index=False)
print("Saved misclassified examples to error_analysis.csv")
