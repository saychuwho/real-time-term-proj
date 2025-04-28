import os
import json
import shutil

annotate_json_file = "./BDD-X-Annotations_v1_cleaned.json"
dataset_path = "/home/saychuwho/BDD/samples-1k/samples-1k/videos/"
dataset_path_2 = "/home/saychuwho/BDD/train/train/videos/"

result_txt = "./check_dataset.json"
copy_path_head = "./videos/"


annotate_dict = {}
with open(annotate_json_file, 'r') as f:
    annotate_dict = json.load(f)

check_dict = {}
for i in range(1, 1001):
    video_path = dataset_path + annotate_dict[str(i)]["Input.Video"] + ".mov"
    copy_path = copy_path_head + annotate_dict[str(i)]["Input.Video"] + ".mov"
    if os.path.exists(video_path):
        print(f"Video {i} exists: {video_path}")
        check_dict[i] = True
        shutil.copy(video_path, copy_path)
        
    else:
        print(f"Video {i} does not exist: {video_path}")
        check_dict[i] = False

for i in range(1001, 7000):
    video_path = dataset_path_2 + annotate_dict[str(i)]["Input.Video"] + ".mov"
    copy_path = copy_path_head + annotate_dict[str(i)]["Input.Video"] + ".mov"
    if os.path.exists(video_path):
        print(f"Video {i} exists: {video_path}")
        check_dict[i] = True
        shutil.copy(video_path, copy_path)
        
    else:
        print(f"Video {i} does not exist: {video_path}")
        check_dict[i] = False

with open(result_txt, 'w') as f:
    json.dump(check_dict, f, indent=4)
