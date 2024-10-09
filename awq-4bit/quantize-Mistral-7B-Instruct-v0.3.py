import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quant_path = "Mistral-7B-Instruct-v0.3-awq-4bit"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load the model
model = AutoAWQForCausalLM.from_pretrained(
    model_name, safetensors=True, device_map="cuda"
)
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Perform the quantization process
# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save the quantized model and tokenizer to the specified path
model.save_quantized("../../Models/" + quant_path)
tokenizer.save_pretrained("../../Models/" + quant_path)
