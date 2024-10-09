# Devices involved
I am testing everithing on my Dell Server with Nvidia Tesla M40 24 Gb (and later on P40 when/if I have it fixed).

# General
All scripts expect that there is a folder called "Models" outside the repo folder where all quantized models
are saved.

# BnB 4 bits
All models in the folder were successfully quantized and tested. 

## Phi-3.5-MoE
`.env.example` file contains one specific line: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It helps to just fit
model into 24 Gb. Without this line quantization and inference crashes.

## CPU inference
Just for gigs I have added CPU inference example `inference-Mistral-7B-Instruct-v0.3-cpu.py`. It's 20x slower on my 10 years old XEONs,
but it's expected. I suspect CPU optimized quatizaton can increase performance dramatically.

# BnB 8 bits
Contains one example of Mixtral 8x7B with CPU offloading to do quantization, but I haven't found the magic combination yet.
I am running it inside a VM that has 40 GB of CPU RAM and so far either it's killed by the OS because the script
occupies all CPU RAM or PyTorch is crashing because 24 GB of GPU RAM is not enough.

# AWQ 4 bits
Refer to [README](awq-4bit/README.txt) about issues that don't allow using it on my hardware.
