#!/bin/bash

# power mode: MAXN, 15W, 30W, 40W / execute with the desired power mode

# python3 src/inference/inference-videollama3.py power_mode MAXN

# python3 src/inference/inference-videollama3.py power_mode 15W

# python3 src/inference/inference-videollama3.py power_mode 30W

# python3 src/inference/inference-videollama3.py power_mode 40W

# output_tokens: 16, 32, 48, 64, 80, 128
# python3 src/inference/inference-videollama3.py output_tokens 4
# python3 src/inference/inference-videollama3.py output_tokens 8
# python3 src/inference/inference-videollama3.py output_tokens 12
# python3 src/inference/inference-videollama3.py output_tokens 16
# python3 src/inference/inference-videollama3.py output_tokens 32
# python3 src/inference/inference-videollama3.py output_tokens 48
# python3 src/inference/inference-videollama3.py output_tokens 64
# # python3 src/inference/inference-videollama3.py output_tokens 80
# # python3 src/inference/inference-videollama3.py output_tokens 128

# input_size: very_short, short, medium, long, very_long
# python3 src/inference/inference-videollama3.py input_size very_short
# python3 src/inference/inference-videollama3.py input_size short
python3 src/inference/inference-videollama3.py input_size medium
python3 src/inference/inference-videollama3.py input_size long
python3 src/inference/inference-videollama3.py input_size very_long

# model_size: 500M, 2.2B, 2B, 7B
# python3 src/inference/inference-SmolLM2.py model_size 500M
# python3 src/inference/inference-SmolLM2.py model_size 2.2B
python3 src/inference/inference-videollama3.py model_size 2B
python3 src/inference/inference-videollama3.py model_size 7B


# python3 ./src/inference/inference-test-videollama3.py
# python ./src/train/train-videollama3.py
# /bin/bash
