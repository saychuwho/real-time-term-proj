import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer, PreTrainedTokenizerBase
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Load your dataset

dataset_path = "./datasets/BDD-X-Annotations-finetune-val.json"

with open(dataset_path, "r") as f:
    raw_data = json.load(f)

# Flatten into (video_path, input_text, target_text) pairs
samples = []
for item in raw_data:
    video_path = item["video"][0]
    convs = item["conversations"]
    # Pair each human prompt with the following gpt answer
    for i in range(0, len(convs) - 1, 2):
        if convs[i]["from"] == "human" and convs[i+1]["from"] == "gpt":
            samples.append({
                "video_path": video_path,
                "input_text": convs[i]["value"],
                "target_text": convs[i+1]["value"]
            })


# Convert to Hugging Face Dataset
dataset = Dataset.from_list(samples)


def preprocess(example):
    try:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": example["video_path"], "fps": 1, "max_frames": 180}},
                    {"type": "text", "text": example["input_text"]},
                ]
            },
        ]
        # Tokenize the input (prompt + video)
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # Tokenize the expected output (target/answer)
        labels = processor.tokenizer(
            example["target_text"],
            return_tensors="pt",
            padding="max_length",
            max_length=inputs["input_ids"].shape[1],  # match input length
            truncation=True
        )["input_ids"].squeeze(0)

        # Flatten tensor outputs for Trainer compatibility
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["labels"] = labels
        return inputs
    except Exception as e:
        print(f"Skipping sample due to error: {e} (video: {example['video_path']})")
        return None

# When mapping, remove None results
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
).filter(lambda x: x is not None)


training_args = TrainingArguments(
    output_dir="./videollama3-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()