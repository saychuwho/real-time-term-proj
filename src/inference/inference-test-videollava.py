import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Quantization config using BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                # Set to True for 4-bit quantization
    bnb_4bit_use_double_quant=True,   # Use double quantization for better accuracy
    bnb_4bit_quant_type="nf4",        # "nf4" is recommended, or use "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype (can use torch.float16 or torch.bfloat16)
)

device = "cuda:0"

processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    device_map={"": device},
    quantization_config=quant_config,
)

video_path = "datasets/inference-test/basketball.mp4"
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

prompt = "USER: <video>\nExplain the video ASSISTANT:"

# inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)
inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=60)

response = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(response)