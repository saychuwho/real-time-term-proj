import json
from scipy.stats import kurtosis
from scipy.stats import iqr
import numpy as np
import matplotlib.pyplot as plt
import os


def data_analysis(data):
    return_dict = {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'max': np.max(data),
        'min': np.min(data),
        'kurtosis': kurtosis(data, fisher=True, bias=False),
        'iqrn': iqr(data, rng=(25, 75), interpolation='linear') / np.mean(data),
    }
    return return_dict

def plot_histogram(data, title, xlabel, ylabel, save_path='histogram.png'):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.close()


test_modes = {
    "power_mode": ["MAXN", "15W", "30W", "40W"],
    "model_size": ["500M", "2.2B", "2B", "7B"],
    "output_tokens": ["4", "8", "12", "16", "32", "48", "64", "128"],
    "input_size": ["very_short", "short", "medium", "long", "very_long"],
}

test_environments = ["desktop", "jetson", "server"]

save_dict = {}

for test_env in test_environments:
    for test_mode in test_modes.keys():
        for test_args in test_modes[test_mode]:
            results_json = f"results-{test_env}/videollama3-results-{test_mode}-{test_args}.json"

            print(f"Processing {results_json}...")

            if not os.path.exists(results_json):
                print(f">> File {results_json} does not exist. Skipping...")
                continue

            with open(results_json, 'r') as f:
                results = json.load(f)

            input_preprocessing_time_list = np.array([t['input_preprocessing_time'] for t in results])
            model_inference_time_list = np.array([t['model_inference_time'] for t in results])




            input_preprocessing_analysis = data_analysis(input_preprocessing_time_list)
            save_path_1 = f"evaluation-results/input_preprocessing_time_analysis_{test_env}_{test_mode}_{test_args}.png"
            plot_histogram(input_preprocessing_time_list, 
                        f'Input Preprocessing Time Distribution {test_env} {test_mode} {test_args}', 
                        'Time (seconds)', 
                        'Frequency', 
                        save_path=save_path_1)


            model_inference_analysis = data_analysis(model_inference_time_list)
            save_path_2 = f"evaluation-results/model_inference_time_analysis_{test_env}_{test_mode}_{test_args}.png"
            plot_histogram(model_inference_time_list, 
                        f'Model Inference Time Distribution {test_env} {test_mode} {test_args}', 
                        'Time (seconds)', 
                        'Frequency', 
                        save_path=save_path_2)

            
            save_dict[f"{test_env}-{test_mode}-{test_args}"] = {
                'input_preprocessing_analysis': input_preprocessing_analysis,
                'model_inference_analysis': model_inference_analysis,
            }


save_json = "evaluation-results/result.json"
with open(save_json, 'w') as f:
    json.dump(save_dict, f, indent=4)