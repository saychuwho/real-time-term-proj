import json
import os
from tqdm import tqdm

annotate_json_file = "./BDD-X-Annotations_v1_cleaned_v2.json"
annotate_dict = {}

with open(annotate_json_file, 'r') as f:
    annotate_dict = json.load(f)

start_point = 1
end_point = 7000

no_exist_list = []
no_exist_list_file = "./no-exist-list.txt"

for i in tqdm(range(start_point, end_point)):
    
    if annotate_dict.get(str(i)) is None: continue

    video_tag = annotate_dict[str(i)]["Input.Video"]
    answer_length = annotate_dict[str(i)]["Answer-Length"]
    for j in range(1, answer_length+1):
        video_after_path = f"./videos-cut/{video_tag}-{j}.mp4"
        
        if not os.path.isfile(video_after_path):
            no_exist_list.append(video_after_path)
    
if not no_exist_list: print("All video splited!")
else: 
    with open(no_exist_list_file, 'w') as f:
        f.write(str(no_exist_list))