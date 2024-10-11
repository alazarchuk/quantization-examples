import torch
import wandb
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Define the model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

run = wandb.init(
    project=f'{os.environ["WANDB_PROJECT_PREFIX"]} -> {model_name.split("/")[1]}',
    job_type="inference",
)

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda", torch_dtype=torch.float16
)

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
