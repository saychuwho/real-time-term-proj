import json
from tqdm import tqdm

annotated_bddx = {}
annotated_file = "./datasets/BDD-X-Annotations_v1_cleaned_v2.json"

annotated_train = []
annotated_val = []
annotated_test = []

annotated_training_file = "./datasets/BDD-X-Annotations-finetune-train.json"
annotated_val_file = "./datasets/BDD-X-Annotations-finetune-val.json"
annotated_test_file = "./datasets/BDD-X-Annotations-finetune-test.json"


with open(annotated_file, 'r') as f: annotated_bddx = json.load(f)

start_point = 1
end_point = 7000

video_path_head = "./datasets/videos-cut/"

tmp_counter = 0

for i in tqdm(range(start_point, end_point)):
    if annotated_bddx.get(str(i)) is None: continue

    answer_length = annotated_bddx[str(i)]["Answer-Length"]
    video_tag = annotated_bddx[str(i)]["Input.Video"]

    tmp_list = []
    for j in range(1, answer_length+1):
        tmp_dict = {}
        tmp_dict["video"] = [ f"{video_path_head}{video_tag}-{j}.mp4" ]
        conversation_list = [
            {
                "from": "human",
                "value": "What is the motion of the ego vehicle?"           
            },
            {
                "from": "gpt",
                "value": annotated_bddx[str(i)][f"Answer-{j}"][str(2)]
            },
            {
                "from": "human",
                "value": "Why does an ego vehicle do this motion?"
            },
            {
                "from": "gpt",
                "value": annotated_bddx[str(i)][f"Answer-{j}"][str(3)]
            }
        ]
        tmp_dict["conversations"] = conversation_list
        if 1 <= i <= 6000: annotated_train.append(tmp_dict)
        elif 6001 <= i < 6300: annotated_val.append(tmp_dict)
        elif 6300 <= i < 7000: annotated_test.append(tmp_dict)


with open(annotated_training_file, 'w') as f: json.dump(annotated_train, f, indent=4)
with open(annotated_val_file, 'w') as f: json.dump(annotated_val, f, indent=4)
with open(annotated_test_file, 'w') as f: json.dump(annotated_test, f, indent=4)
print(f"Annotated train: {len(annotated_train)}")
print(f"Annotated val: {len(annotated_val)}")
print(f"Annotated test: {len(annotated_test)}")