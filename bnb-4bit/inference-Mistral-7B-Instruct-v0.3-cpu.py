from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

input_text = "Write me a story about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
