import torch
import wandb
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Define the model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

run = wandb.init(
    project=f'{os.environ["WANDB_PROJECT_PREFIX"]} -> {model_name.split("/")[1]} (cpu)',
    job_type="inference",
)

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
