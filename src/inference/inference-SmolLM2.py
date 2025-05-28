from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
from tqdm import tqdm
import sys
import time

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
processor = AutoProcessor.from_pretrained(preproc_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")


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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": text_prompt}
            ]
        },
    ]

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "path": video_path},
    #             {"type": "text", "text": "Explain this video"}
    #         ]
    #     },
    # ]

    # input preprocessing
    start_time = time.perf_counter()
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    input_preprocessing_time = end_time - start_time


    # model inference time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    
    elapsed_time += end_time - start_time
    model_inference_time = end_time - start_time


    # output postprocessing
    start_time = time.perf_counter()

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time
    output_postprocessing_time = end_time - start_time

    elapsed_times = (input_preprocessing_time, model_inference_time, output_postprocessing_time)

    # debug
    prompt = processor.decode(generated_ids[0])

    # return generated_texts[0], elapsed_time, elapsed_times
    return generated_texts[0], elapsed_time, elapsed_times, prompt


qa_results = []

annotations = annotations[:200]

counter = 0
for conversation in tqdm(annotations, desc="Processing conversations"):
    video_path = conversation['video'][0]
    
    for i in [0,2]:
        text_prompt = conversation['conversations'][i]["value"]
        
        # response, elapsed_time, elapsed_times = get_response(text_prompt, video_path, output_tokens)
        response, elapsed_time, elapsed_times, prompt = get_response(text_prompt, video_path, output_tokens)
        
        qa_results.append({
            "video_path": video_path,
            "text_prompt": text_prompt,
            "response": response,
            "ground_truth": conversation['conversations'][i + 1]["value"],
            "elapsed_time": elapsed_time,
            "input_preprocessing_time": elapsed_times[0],
            "model_inference_time": elapsed_times[1],
            "output_postprocessing_time": elapsed_times[2], 
            "prompt": prompt,
        })

    if counter % 50 == 0:
        save_json = f"results/SmolLM2-results-{test_mode}-{test_args}-{counter}.json"
        with open(save_json, 'w') as f:
            json.dump(qa_results, f, indent=4)

    counter += 1

save_json = f"results/SmolLM2-results-{test_mode}-{test_args}.json"

with open(save_json, 'w') as f:
    json.dump(qa_results, f, indent=4)

print(f"Results saved to {save_json}")