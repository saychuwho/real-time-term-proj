import torch
import json
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import subprocess
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False,
        init_lora_weights="gaussian"
    )
lora_config.inference_mode = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
    device_map="auto"
)
model.add_adapter(lora_config)
model.enable_adapters()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

processor = AutoProcessor.from_pretrained(model_name)

# Load your dataset
dataset_path = "./datasets/BDD-X-Annotations-finetune-train.json"

# dataset_path_2 = "./datasets/BDD-X-Annotations-finetune-sample-traffic.json"

with open(dataset_path, "r") as f:
    raw_data = json.load(f)

# with open(dataset_path_2, "r") as f:
#     raw_data_2 = json.load(f)

skip_list = [
    "./datasets/videos-cut/00495359-f020db82-7.mp4"
    "./datasets/videos-cut/05fa6494-2fad5381-2.mp4",
    "./datasets/videos-cut/03f6e688-415b77da-2.mp4",
    "./datasets/videos-cut/036f3b5f-ee1521de-6.mp4",
    "./datasets/videos-cut/02d478d1-cfb83eb8-5.mp4",
    "./datasets/videos-cut/00268999-0b20ef00-7.mp4",
    "./datasets/videos-cut/038db692-a3659ab7-1.mp4",
    "./datasets/videos-cut/06257efc-a0ad911e-3.mp4",
    "./datasets/videos-cut/05e5b952-70bc5050-6.mp4",
    "./datasets/videos-cut/02d478d1-e6811391-9.mp4",
    "./datasets/videos-cut/01d25bbc-670b6caa-6.mp4",
    "./datasets/videos-cut/06fce5b8-efc1a0dd-8.mp4",
    "./datasets/videos-cut/00810e80-37641274-2.mp4",
    "./datasets/videos-cut/022ba62d-3353f550-6.mp4",
    "./datasets/videos-cut/0714456d-083ade57-5.mp4",
    "./datasets/videos-cut/066d6f37-6037f6cf-8.mp4",
    "./datasets/videos-cut/0336de82-802d2437-2.mp4",
    "./datasets/videos-cut/06cb522f-f1ff4334-5.mp4",
    "./datasets/videos-cut/05e261cb-ef8cd52f-6.mp4",
    "./datasets/videos-cut/03d34d05-4a61174c-3.mp4",
    "./datasets/videos-cut/06cb522f-f3b53ccf-7.mp4",
    "./datasets/videos-cut/018d962a-fd869a57-6.mp4",
    "./datasets/videos-cut/07cacc74-61869ca0-6.mp4"
]

def is_video_valid(video_path):
    """Returns True if ffprobe can read the video file, False otherwise."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        return result.returncode == 0 and result.stdout.strip() != b""
    except Exception as e:
        print(f"ffprobe failed for {video_path}: {e}")
        return False

# Flatten into (video_path, input_text, target_text) pairs
print(f"Loading samples... {len(raw_data)}")


samples = []

# for item, item2 in zip(raw_data, raw_data_2):
for item in raw_data:
    #  print(item)
    video_path = item["video"][0]

    if video_path in skip_list: continue
    
    if not os.path.exists(video_path): continue

    if not is_video_valid(video_path): continue

    convs = item["conversations"]

    # Pair each human prompt with the following gpt answer
    for i in range(0, len(convs) - 1, 2):
        if convs[i]["from"] == "human" and convs[i+1]["from"] == "gpt":
            samples.append({
                "video_path": video_path,
                "input_text": convs[i]["value"],
                "target_text": convs[i+1]["value"]
            })
    # convs2 = item2["conversations"]    
    # for i in range(0, len(convs2) - 1, 2):
    #     if convs2[i]["from"] == "human" and convs2[i+1]["from"] == "gpt":
    #         samples.append({
    #             "video_path": video_path,
    #             "input_text": convs2[i]["value"],
    #             "target_text": convs2[i+1]["value"]
    #         })


print(f"Loaded {len(samples)} samples.")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(samples)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

def collate_fn(examples):
    instances = []
    for example in examples:
        prompt = example["target_text"]

        user_content = [{"type": "text", "text": example["input_text"]}]
        user_content.append({"type": "video", "path": example["video_path"]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
        instances.append(instance)


    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    )

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out


model_name = model_name.split("/")[-1]

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=1,
    optim="adamw_torch", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
    bf16=True,
    output_dir=f"./finetuned-model/smollm2-train",
    hub_model_id=f"{model_name}-video-feedback",
    remove_unused_columns=False,
    # report_to="tensorboard",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset,
)

torch.cuda.empty_cache()

print("Training...")
trainer.train()