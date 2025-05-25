import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer, PreTrainedTokenizerBase
from datasets import Dataset

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    device_map={"": device}
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Load your dataset

dataset_path = "./datasets/BDD-X-Annotations-finetune-sample-traffic.json"
dataset_path_2 = "./datasets/BDD-X-Annotations-finetune-sample.json"

exceptions_file = "src/exceptions-videollama3-2.txt"

with open(dataset_path, "r") as f:
    raw_data = json.load(f)

with open(dataset_path_2, "r") as f:
    raw_data_2 = json.load(f)

# Flatten into (video_path, input_text, target_text) pairs
print("Loading samples...")
samples = []
for item, item2 in tqdm(zip(raw_data, raw_data_2)):
    video_path = item["video"][0]
    convs = item["conversations"]
    convs2 = item2["conversations"]
    # Pair each human prompt with the following gpt answer
    for i in range(0, len(convs) - 1, 2):
        if convs[i]["from"] == "human" and convs[i+1]["from"] == "gpt":
            samples.append({
                "video_path": video_path,
                "input_text": convs[i]["value"],
                "target_text": convs[i+1]["value"]
            })
    for i in range(0, len(convs2) - 1, 2):
        if convs2[i]["from"] == "human" and convs2[i+1]["from"] == "gpt":
            samples.append({
                "video_path": video_path,
                "input_text": convs2[i]["value"],
                "target_text": convs2[i+1]["value"]
            })


# Convert to Hugging Face Dataset
dataset = Dataset.from_list(samples)

exceptions = []

def preprocess(example):
    try:
        # Build the full prompt + answer as a single sequence
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": example["video_path"], "fps": 1, "max_frames": 64}},
                    {"type": "text", "text": example["input_text"]},
                ]
            },
            {
                "role": "assistant",
                "content": example["target_text"]
            }
        ]
        # Tokenize the full sequence
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask out the prompt part with -100
        # Find where the answer starts
        prompt_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": example["video_path"], "fps": 1, "max_frames": 64}},
                    {"type": "text", "text": example["input_text"]},
                ]
            }
        ]
        prompt_inputs = processor(
            conversation=prompt_conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask out the prompt part

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    except Exception as e:
        print(f"Skipping sample due to error: {e} (video: {example['video_path']})")
        #  exceptions.append(f"{example['video_path']}")
        return {"input_ids": None, "attention_mask": None, "labels": None}


# When mapping, remove None results
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=24
).filter(lambda x: x["input_ids"] is not None and x["labels"] is not None)


# with open(exceptions_file, 'w') as f:
#     for exception in exceptions:
#         f.write(f"{exception}\n")

print(f"Number of valid samples: {len(tokenized_dataset)}")

training_args = TrainingArguments(
    output_dir="/workspaces/finetuned-model/videollama3-finetuned-sample-traffic",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=6000,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

torch.cuda.empty_cache()

trainer.train()