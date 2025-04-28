from moviepy.editor import *
import json
import sys


annotate_json_file = "./datasets/BDD-X-Annotations_v1_cleaned_v2.json"
annotate_dict = {}

with open(annotate_json_file, 'r') as f:
    annotate_dict = json.load(f)


# if len(sys.argv) < 2:
#     print("E: put start point")
#     sys.exit()

# start_point = int(sys.argv[1])

# if len(sys.argv) == 2: end_point = start_point + 1000
# else: end_point = int(sys.argv[2])

start_point = 1
end_point = 7000

for i in range(start_point, end_point):
    print(f"##### Processing {i} #####")
    if annotate_dict.get(str(i)) is None: continue

    video_tag = annotate_dict[str(i)]["Input.Video"]
    video_path = f"./videos/{video_tag}.mov"
    clip = VideoFileClip(video_path)

    answer_length = annotate_dict[str(i)]["Answer-Length"]
    for j in range(1, answer_length+1):
        print(f"\n##### Processing {i}-Answer-{j} #####")
        video_after_path = f"./videos-cut/{video_tag}-{j}.mp4"
        t_start = annotate_dict[str(i)][f"Answer-{j}"]["0"]
        t_end = annotate_dict[str(i)][f"Answer-{j}"]["1"]

        answer_clip = clip.subclip(t_start,t_end)
        answer_clip = answer_clip.resize(newsize=(1280,720))
        answer_clip.write_videofile(video_after_path, 
                             codec="libx264", 
                             audio_codec="aac")
