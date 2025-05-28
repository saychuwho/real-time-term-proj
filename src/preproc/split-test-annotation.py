import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

test_annotation_json = 'datasets/BDD-X-Annotations-finetune-test.json'

with open(test_annotation_json, 'r') as f:
    test_annotations = json.load(f)

video_lengths = []
video_lengths_dict = {
    'very_short': [],
    'short': [],
    'medium': [],
    'long': [],
    'very_long': []
}

for annotation in tqdm(test_annotations):
    video_path = annotation['video'][0]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    
    annotation['duration'] = duration  

    if 1 <= duration <= 3: video_lengths_dict['very_short'].append(annotation)
    elif 3 < duration <= 6: video_lengths_dict['short'].append(annotation)
    elif 6 < duration <= 10: video_lengths_dict['medium'].append(annotation)
    elif 10 < duration <= 20: video_lengths_dict['long'].append(annotation)
    elif duration > 20: video_lengths_dict['very_long'].append(annotation)
    
    video_lengths.append(duration)
    cap.release()

# Save the updated annotations with durations
for key, value in video_lengths_dict.items():
    print(f"Number of videos in {key} category: {len(value)}")
    with open(f'datasets/BDD-X-Annotations-finetune-test-{key}.json', 'w') as f:
        json.dump(value, f, indent=4)


# Calculate the average video length
average_length = np.mean(video_lengths)
print(f"Average video length: {average_length:.2f} seconds")

# Plot histogram of video lengths with custom x-ticks
num_bins = 40
plt.hist(video_lengths, bins=num_bins, color='skyblue', edgecolor='black')
plt.xlabel('Video Length (seconds)')
plt.ylabel('Count')
plt.title('Histogram of Video Lengths')

# Set x-ticks at every integer second within the range of your data
min_length = int(np.floor(min(video_lengths)))
max_length = int(np.ceil(max(video_lengths)))
plt.xticks(np.arange(min_length, max_length + 1, 1), rotation=45)

plt.tight_layout()
plt.savefig('datasets/test_video_lengths_histogram.png')