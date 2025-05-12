import json
from tqdm import tqdm

annotated_bddx = {}
annotated_file = "./datasets/BDD-X-Annotations_v1_cleaned_v2.json"

annotated_videollama = []
annotated_videollama_file = "./datasets/BDD-X-Annotations-finetune.json"


with open(annotated_file, 'r') as f: annotated_bddx = json.load(f)

start_point = 1
end_point = 7000

video_path_head = "./datasets/videos-cut/"

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
        annotated_videollama.append(tmp_dict)

with open(annotated_videollama_file, 'w') as f: json.dump(annotated_videollama, f, indent=4)
