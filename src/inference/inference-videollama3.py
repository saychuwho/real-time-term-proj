import torch
import sys
import json
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


if len(sys.argv) < 2:
    print("Usage: python inference-SmolLM2.py <test_mode> <test_args>")
    sys.exit(1)

test_mode = sys.argv[1]
test_args = sys.argv[2]

option_json = "src/inference/SmolLM2-inference-option.json"

with open(option_json, 'r') as f:
    options = json.load(f)

model_path = options[test_mode][test_args]["model_path"]
preproc_path = options[test_mode][test_args]["preproc_path"]
output_tokens = options[test_mode][test_args]["output_tokens"]
model_quantization_level = options[test_mode][test_args]["model_quantization_level"]
input_size = options[test_mode][test_args]["input_size"]


# load model and processor
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(preproc_path, trust_remote_code=True)


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
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": text_prompt},
            ]
        },
    ]

    # input preprocessing
    start_time = time.perf_counter()
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    input_preprocessing_time = end_time - start_time

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # model inference time

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
    start_time = time.perf_counter()
    
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
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
