# General
All scripts expect that there is a folder called "Models" outside the repo folder where all quantized models
are saved.

# BnB 4 bits
All models in the folder were successfully quantized though none of them were tested.
I am planning to test Inference on Tesla M40 in the nearest future. While I have certain
confidence in almost all models, I don't know how Phi MoE will behave, because it requires
Flash Attention library. Everything I know about it is that it speeds up performance on modern
Nvidia architectures like Ada, but I don't know if it's compatible with dinosaurs like my M40 or P40.

# BnB 8 bits
Contains one example of Mixtral 8x7B with CPU offloading to do quantization, but I haven't found the magic combination yet.
I am running it inside a VM that has 40 GB of CPU RAM and so far either it's killed by the OS because the script
occupies all CPU RAM or PyTorch is crashing because 24 GB of GPU RAM is not enough.
