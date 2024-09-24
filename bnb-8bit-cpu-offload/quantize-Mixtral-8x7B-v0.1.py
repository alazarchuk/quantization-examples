import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Define the model name and the path where the quantized model will be saved
model_name = "mistralai/Mixtral-8x7B-v0.1"
quant_path = "Mixtral-8x7B-v0.1-bnb-8bit"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the quantization settings for loading the model in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for 8-bit quantization
)

device_map = {
    # "transformer.word_embeddings": 0,
    # "transformer.word_embeddings_layernorm": 0,
    # "lm_head": "cpu",
    # "transformer.h": 0,
    # "transformer.ln_f": 0,
    "model.embed_tokens.weight": 0,
    "model.layers.0": "cpu",
    "model.layers.1": "cpu",
    "model.layers.2": "cpu",
    "model.layers.3": "cpu",
    "model.layers.4": "cpu",
    "model.layers.5": "cpu",
    "model.layers.6": "cpu",
    "model.layers.7": "cpu",
    "model.layers.8": "cpu",
    "model.layers.9": "cpu",
    "model.layers.10": "cpu",
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": 0,
    "model.layers.15": 0,
    "model.layers.16": 0,
    "model.layers.17": 0,
    "model.layers.18": 0,
    "model.layers.19": 0,
    "model.layers.20": 0,
    "model.layers.21": 0,
    "model.layers.22": 0,
    "model.layers.23": 0,
    "model.layers.24": 0,
    "model.layers.25": 0,
    "model.layers.26": 0,
    "model.layers.27": 0,
    "model.layers.28": 0,
    "model.layers.29": 0,
    "model.layers.30": 0,
    "model.layers.31": 0,
    # "model.layers.0.block_sparse_moe.gate.weight": 0,
    # "model.layers.0.input_layernorm.weight": 0,
    # "model.layers.0.post_attention_layernorm.weight": 0,
    # "model.layers.0.self_attn.k_proj.weight": 0,
    # "model.layers.0.self_attn.o_proj.weight": 0,
    # "model.layers.0.self_attn.q_proj.weight": 0,
    # "model.layers.0.self_attn.v_proj.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.0.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.0.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.0.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.0.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.1.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.1.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.1.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.1.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.2.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.2.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.2.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.2.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.3.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.3.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.3.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.3.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.4.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.4.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.4.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.4.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.5.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.5.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.5.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.5.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.6.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.6.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.6.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.6.w4.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.7.w1.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.7.w2.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.7.w3.weight": 0,
    # "model.layers.0.block_sparse_moe.experts.7.w4.weight": 0,
}

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map=device_map
)

# Save the quantized model and tokenizer to the specified directory
model.save_pretrained("../../Models/" + quant_path, safetensors=True)
tokenizer.save_pretrained("../../Models/" + quant_path)
