import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
quant_path = "Mistral-Nemo-Instruct-2407-bnb-4bit"
model_name = "../../Models/" + quant_path

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
