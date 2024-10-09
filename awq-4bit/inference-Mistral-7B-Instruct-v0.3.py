import torch
import wandb
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM


load_dotenv()

# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
quant_path = "Mistral-7B-Instruct-v0.3-awq-4bit"
model_name = "../../Models/" + quant_path

run = wandb.init(
    project=f'{os.environ["WANDB_PROJECT_PREFIX"]} -> {quant_path}',
    job_type="inference",
)

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with the specified quantization configuration
model = AutoAWQForCausalLM.from_pretrained(model_name, device_map="cuda")

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
