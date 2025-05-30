import json
import pandas as pd

test_modes = {
    "power_mode": ["MAXN", "15W", "30W", "40W"],
    "model_size": ["500M", "2.2B", "2B", "7B"],
    "output_tokens": ["4", "8", "12", "16", "32"],
    "input_size": ["very_short", "short", "medium", "long", "very_long"],
}

test_environments = ["desktop", "jetson", "server"]

result_json_file = "evaluation-results/result.json"

with open(result_json_file, 'r') as f:
    results = json.load(f)

for test_mode in test_modes.keys():
    df_setting = {"test_env": []}
    for test_args in test_modes[test_mode]: df_setting[test_args] = []
    df_inference = pd.DataFrame(df_setting)
    df_preprocessing = pd.DataFrame(df_setting)
    
    df_inference_kurtosis = pd.DataFrame(df_setting)
    df_preprocessing_kurtosis = pd.DataFrame(df_setting)
    
    df_inference_iqrn = pd.DataFrame(df_setting)
    df_preprocessing_iqrn = pd.DataFrame(df_setting)

    df_inference_max = pd.DataFrame(df_setting)
    df_preprocessing_max = pd.DataFrame(df_setting)

    for test_env in test_environments:
        new_row_inference = {"test_env": test_env}
        new_row_preprocessing = {"test_env": test_env}

        new_row_inference_kurtosis = {"test_env": test_env}
        new_row_preprocessing_kurtosis = {"test_env": test_env}
        
        new_row_inference_iqrn = {"test_env": test_env}
        new_row_preprocessing_iqrn = {"test_env": test_env}

        new_row_inference_max = {"test_env": test_env}
        new_row_preprocessing_max = {"test_env": test_env}

        for test_args in test_modes[test_mode]:
            key = f"{test_env}-{test_mode}-{test_args}"
            if key in results:
                new_row_inference[test_args] = results[key]['model_inference_analysis']['mean']
                new_row_preprocessing[test_args] = results[key]['input_preprocessing_analysis']['mean']

                new_row_inference_kurtosis[test_args] = results[key]['model_inference_analysis']['kurtosis']
                new_row_preprocessing_kurtosis[test_args] = results[key]['input_preprocessing_analysis']['kurtosis']
                
                new_row_inference_iqrn[test_args] = results[key]['model_inference_analysis']['iqrn']
                new_row_preprocessing_iqrn[test_args] = results[key]['input_preprocessing_analysis']['iqrn']

                new_row_inference_max[test_args] = results[key]['model_inference_analysis']['max']
                new_row_preprocessing_max[test_args] = results[key]['input_preprocessing_analysis']['max']
            else:
                new_row_inference[test_args] = None
                new_row_preprocessing[test_args] = None

                new_row_inference_kurtosis[test_args] = None
                new_row_preprocessing_kurtosis[test_args] = None

                new_row_inference_iqrn[test_args] = None
                new_row_preprocessing_iqrn[test_args] = None

                new_row_inference_max[test_args] = None
                new_row_preprocessing_max[test_args] = None

        df_inference.loc[len(df_inference)] = new_row_inference
        df_preprocessing.loc[len(df_preprocessing)] = new_row_preprocessing
        
        df_inference_kurtosis.loc[len(df_inference_kurtosis)] = new_row_inference_kurtosis
        df_preprocessing_kurtosis.loc[len(df_preprocessing_kurtosis)] = new_row_preprocessing_kurtosis
        
        df_inference_iqrn.loc[len(df_inference_iqrn)] = new_row_inference_iqrn
        df_preprocessing_iqrn.loc[len(df_preprocessing_iqrn)] = new_row_preprocessing_iqrn

        df_inference_max.loc[len(df_inference_max)] = new_row_inference_max
        df_preprocessing_max.loc[len(df_preprocessing_max)] = new_row_preprocessing_max

    df_inference.to_csv(f"evaluation-results/inference_time_{test_mode}.csv", index=False)
    df_preprocessing.to_csv(f"evaluation-results/preprocessing_time_{test_mode}.csv", index=False)
    
    df_inference_kurtosis.to_csv(f"evaluation-results/inference_time_kurtosis_{test_mode}.csv", index=False)
    df_preprocessing_kurtosis.to_csv(f"evaluation-results/preprocessing_time_kurtosis_{test_mode}.csv", index=False)
    
    df_inference_iqrn.to_csv(f"evaluation-results/inference_time_iqrn_{test_mode}.csv", index=False)
    df_preprocessing_iqrn.to_csv(f"evaluation-results/preprocessing_time_iqrn_{test_mode}.csv", index=False)
    
    df_inference_max.to_csv(f"evaluation-results/inference_time_max_{test_mode}.csv", index=False)
    df_preprocessing_max.to_csv(f"evaluation-results/preprocessing_time_max_{test_mode}.csv", index=False)

    print(f"Saved results for {test_mode} to CSV files.")
