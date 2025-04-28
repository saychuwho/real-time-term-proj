import pandas as pd
import math
import json
from urllib.parse import urlparse
import os

# clean original annotation file
df = pd.read_csv("./datasets/BDD-X-Annotations_v1.csv")
df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')
df.to_csv("./datasets/BDD-X-Annotations_v1_cleaned.csv", index=False)

# make cleaned annotation file into json format
annotation_cleaned_file = "./datasets/BDD-X-Annotations_v1_cleaned.csv"
json_file = "./datasets/BDD-X-Annotations_v1_cleaned_v2.json"

df = pd.read_csv(annotation_cleaned_file)
df_col = df.columns.tolist()

dump_dict = {}

for index, row in df.iterrows():
    row_list = row.tolist()
    dump_dict[index+1] = {}
    tmp_list = []

    # Write Input.Video
    path = urlparse(row_list[0]).path
    filename = os.path.basename(path)
    basename, _ = os.path.splitext(filename)
    dump_dict[index+1][df_col[0]] = basename

    
    # Write Answer labels
    row_list_index = 1
    valid_length = 0
    for i in range(1, 15):
        tmp_dict = {}
        tmp_dict_valid = True
        
        if type(row_list[row_list_index]) == float and math.isnan(row_list[row_list_index]): break

        for j in range(4):
            tmp_value = row_list[row_list_index]
            if type(tmp_value) == float and math.isnan(tmp_value):
                tmp_dict_valid = False
                break

            # turn "0" into 0
            if j == 0:
                if tmp_value == 'X':
                    tmp_dict_valid = False
                    break
                tmp_value = int(tmp_value)
            if j == 1:
                tmp_value = int(tmp_value)

            tmp_dict[j] = tmp_value
            row_list_index += 1
        
        if tmp_dict_valid: 
            valid_length += 1
            dump_dict[index+1][f"Answer-{valid_length}"] = tmp_dict
    
    dump_dict[index+1]["Answer-Length"] = valid_length

    
with open(json_file, 'w') as f:
    json.dump(dump_dict, f, indent=4)
