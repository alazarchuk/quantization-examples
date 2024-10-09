import wandb
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Define the model name and the path where the quantized model will be saved
quant_path = "gemma-2-9b-it-bnb-4bit"
model_name = "../../Models/" + quant_path

run = wandb.init(
    project=f'{os.environ["WANDB_PROJECT_PREFIX"]} -> {quant_path}',
    job_type="inference",
)

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
