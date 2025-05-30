#!bin/bash

# python3 src/inference/inference-test-videollava.py

#!/bin/bash

# power mode: MAXN, 15W, 30W, 40W / execute with the desired power mode

# python3 src/inference/inference-videollava.py power_mode MAXN

# python3 src/inference/inference-videollava.py power_mode 15W

python3 src/inference/inference-videollava.py power_mode 30W

# python3 src/inference/inference-videollava.py power_mode 40W

# output_tokens: 16, 32, 48, 64, 80, 128
python3 src/inference/inference-videollava.py output_tokens 4
python3 src/inference/inference-videollava.py output_tokens 8
python3 src/inference/inference-videollava.py output_tokens 12
python3 src/inference/inference-videollava.py output_tokens 16
python3 src/inference/inference-videollava.py output_tokens 32
python3 src/inference/inference-videollava.py output_tokens 48
python3 src/inference/inference-videollava.py output_tokens 64
python3 src/inference/inference-videollava.py output_tokens 80
python3 src/inference/inference-videollava.py output_tokens 128

# input_size: very_short, short, medium, long, very_long
python3 src/inference/inference-videollava.py input_size very_short
python3 src/inference/inference-videollava.py input_size short
python3 src/inference/inference-videollava.py input_size medium
python3 src/inference/inference-videollava.py input_size long
python3 src/inference/inference-videollava.py input_size very_long

# /bin/bash

