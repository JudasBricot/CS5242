from pathlib import Path
import pandas as pd
import pickle
import os

if __name__=="__main__":
    evaluation_folders = [
        "/workspace/output/person_1_alt_reg_2_gpu_longer",
        "/workspace/output/person_1_reg_2_gpu_longer",
        "/workspace/output/person_1_no_reg",
        "/workspace/output/person_1_reg_2_gpu_longer_cr",
        "/workspace/output/person_2_alt_reg_2_gpu_longer",
        "/workspace/output/person_2_reg_2_gpu_longer",
        "/workspace/output/person_2_no_reg",
        "/workspace/output/person_2_reg_2_gpu_longer_cr",
        "/workspace/output/person_3_alt_reg_2_gpu_longer",
        "/workspace/output/person_3_reg_2_gpu_longer",
        "/workspace/output/person_3_no_reg",
        "/workspace/output/person_3_reg_2_gpu_longer_cr"
    ]

    eventual_df = pd.DataFrame()
    for eval_pkl in evaluation_folders:
        #get the instance
        instance_eval_dir = Path(eval_pkl)
        instance_name = "_".join(instance_eval_dir.name.split('_')[:2])
        setup = "_".join(instance_eval_dir.name.split('_')[2:])

        filename = os.path.join(eval_pkl, "evaluate.pkl")
        with open(filename, "rb") as f:
            row_pkl = pickle.load(f)
        #add in new column for the data 
        row_pkl['data name'] = instance_name
        row_pkl['setup'] = setup
        eventual_df = pd.concat([eventual_df, row_pkl])

    

    eventual_df.to_csv("merged_results.csv")
    