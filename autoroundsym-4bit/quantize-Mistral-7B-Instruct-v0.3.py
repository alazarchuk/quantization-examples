import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound

# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quant_path = "Mistral-7B-Instruct-v0.3-autoroundsym-4bit"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the number of bits for quantization
bits = 4

# Define the group size for quantization
group_size = 128

# Define whether to use symmetric quantization
sym = True

# Define the batch size for quantization
batch_size = 2

# Define the sequence length for quantization
seqlen = 512

# Define the device to be used for quantization
device = "cuda"

# Initialize the AutoRound object with the specified parameters
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    batch_size=batch_size,
    seqlen=seqlen,
    sym=sym,
    gradient_accumulate_steps=4,
    device=device,
)

# Perform the quantization process
autoround.quantize()

# Save the quantized model and tokenizer to the specified path
autoround.save_quantized("../../Models/" + quant_path)
tokenizer.save_pretrained("../../Models/" + quant_path)
