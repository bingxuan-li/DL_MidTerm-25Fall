from unsloth import FastLanguageModel
import torch, re, tqdm
import pandas as pd
from datasets import load_dataset

# Define the path where the model checkpoint was saved in Google Drive
save_path = "./math_verifier_model/checkpoint-12000"

# Load the model and tokenizer from the saved path
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Prepare the loaded model for faster inference
FastLanguageModel.for_inference(model)

print(f"Model and tokenizer loaded from: {save_path}")
# Load LoRA adapters

# Load the official test set
test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
predictions = []

# Create the prompt template for inference (no answer included)
inference_prompt = (
    "Decide if the given answer is correct. Respond with True or False only.\n"
    "Question: {}\n"
    "Proposed Answer: {}\n"
    "Solution: {}\n"
    "Correctness: "
)

def clean_solution(example):
    solution = re.sub(r"```.*?```", "", example['solution'], flags=re.DOTALL)
    solution = re.sub(r"\d+\.\d+", lambda x: f"{float(x.group()):.4f}", solution)
    example['solution'] = solution
    example['is_correct'] = str(example['is_correct'])
    return example

test_dataset = test_dataset.map(clean_solution)

# A simple function to parse 'True' or 'False' from the model's raw output
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

# Loop through the test dataset and generate a prediction for each example
for example in tqdm.tqdm(test_dataset):
    question = example["question"]
    solution = example["solution"]
    answer = example["answer"]

    # Format the prompt
    prompt = inference_prompt.format(question, answer, str(solution))
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the prediction
    outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0]

    # Parse the prediction and add it to our list
    prediction = parse_output(response_text, answer = example["answer"])
    predictions.append(prediction)

# Create the submission DataFrame
submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'is_correct': predictions
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

# print("\nSubmission file 'submission.csv' created successfully!")
# print("You can now download this file and submit it to the Kaggle competition.")

# example = test_dataset[0] # You can change the index (e.g., to 1, 2, 50)
# question = example["question"]
# solution = example["solution"]

# # Format the prompt with the validation data
# inputs = tokenizer(
# [
#     inference_prompt.format(question, example["answer"], str(solution))
# ], return_tensors = "pt").to("cuda")

# # Generate the model's response
# outputs = model.generate(**inputs, max_new_tokens = 8, use_cache = True)
# response = tokenizer.batch_decode(outputs)

# # Print the results
# print("#### QUESTION ####")
# print(question)
# print("\n#### SOLUTION ####")
# print(solution)
# print("\n#### MODEL'S PREDICTION ####")
# # We process the output to show only the generated text
# print(response[0])
# print("\n#### CORRECT ANSWER ####")
# print(example["is_correct"])

# prediction = parse_output(response[0], answer = example["answer"])
# print("\n#### PARSED PREDICTION ####")
# print(prediction)