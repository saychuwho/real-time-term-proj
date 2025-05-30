import torch
import sys
import json
import time
from tqdm import tqdm
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig
import av


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


if len(sys.argv) < 2:
    print("Usage: python inference-SmolLM2.py <test_mode> <test_args>")
    sys.exit(1)

test_mode = sys.argv[1]
test_args = sys.argv[2]

option_json = "src/inference/videollava-inference-option.json"

with open(option_json, 'r') as f:
    options = json.load(f)

model_path = options[test_mode][test_args]["model_path"]
preproc_path = options[test_mode][test_args]["preproc_path"]
output_tokens = options[test_mode][test_args]["output_tokens"]
model_quantization_level = options[test_mode][test_args]["model_quantization_level"]
input_size = options[test_mode][test_args]["input_size"]


# load model and processor

# Quantization config using BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                # Set to True for 4-bit quantization
    bnb_4bit_use_double_quant=True,   # Use double quantization for better accuracy
    bnb_4bit_quant_type="nf4",        # "nf4" is recommended, or use "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype (can use torch.float16 or torch.bfloat16)
)

device = "cuda:0"
processor = VideoLlavaProcessor.from_pretrained(model_path)
model = VideoLlavaForConditionalGeneration.from_pretrained(
    model_path,
    device_map={"": device},
    quantization_config=quant_config,
)


# load annotations
annotation_json = f"datasets/BDD-X-Annotations-finetune-test-{input_size}.json"
with open(annotation_json, 'r') as f:
    annotations = json.load(f)


print(f"\ntest_mode: {test_mode}")
print(f"test_args: {test_args}")
print(f"model_path: {model_path}")
print(f"preproc_path: {preproc_path}")
print(f"output_tokens: {output_tokens}")
print(f"model_quantization_level: {model_quantization_level}")
print(f"input_size: {input_size}\n")


def get_response(text_prompt, video_path, max_new_tokens):
    prompt = f"USER: <video>\n{text_prompt} ASSISTANT:"

    # print(f"Processing video: {video_path}")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    # input preprocessing
    # print(f"Processing prompt: {prompt}")

    start_time = time.perf_counter()
    
    inputs = processor(
        text=prompt,
        videos=video,
        return_tensors="pt"
    ).to(device)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    input_preprocessing_time = end_time - start_time

    # model inference time
    # print("Running model inference...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    
    elapsed_time += end_time - start_time
    model_inference_time = end_time - start_time

    # output postprocessing
    # print("Postprocessing output...")
    start_time = time.perf_counter()
    
    response = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    end_time = time.perf_counter()
    
    elapsed_time += end_time - start_time
    output_postprocessing_time = end_time - start_time

    elapsed_times = (input_preprocessing_time, model_inference_time, output_postprocessing_time)

    return response, elapsed_time, elapsed_times


qa_results = []
annotations = annotations[:200]

counter = 0
for conversation in tqdm(annotations, desc="Processing conversations"):
    video_path = conversation['video'][0]

    for i in [0,2]:
        text_prompt = conversation['conversations'][i]["value"]
        
        response, elapsed_time, elapsed_times = get_response(text_prompt, video_path, output_tokens)
        
        qa_results.append({
            "video_path": video_path,
            "text_prompt": text_prompt,
            "response": response,
            "ground_truth": conversation['conversations'][i + 1]["value"],
            "elapsed_time": elapsed_time,
            "input_preprocessing_time": elapsed_times[0],
            "model_inference_time": elapsed_times[1],
            "output_postprocessing_time": elapsed_times[2]
        })

    if counter % 50 == 0:
        save_json = f"results/videollama3-results-{test_mode}-{test_args}-{counter}.json"
        with open(save_json, 'w') as f:
            json.dump(qa_results, f, indent=4)

    counter += 1


save_json = f"results/videollama3-results-{test_mode}-{test_args}.json"

with open(save_json, 'w') as f:
    json.dump(qa_results, f, indent=4)

print(f"Results saved to {save_json}")
