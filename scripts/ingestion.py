from scripts.functions import *

#%% Optional Parameters
''' 
When aggregate dataset is set to True, the script will combine all .txt files in the specified directory into a single Polars DataFrame.
When set to False, the script will not perform this aggregation and will instead read the existing parquet file
'''
aggregate_dataset = True

#%% File Paths + Data Ingestion
'''
Define the base directory and generate the key file paths for the project. 
The file paths are then written to a YAML file for easy reference.
'''
#dir = r"C:\Users\soloe\OneDrive\Documents\Data Projects\Baby Names"
dir = "./"
file_paths = {
        "dir_path": dir,
        "dataset_year_path": clean_file_path(dir, "data/Year", ""),
        "dataset_state_path": clean_file_path(dir, "data/State", ""),
        "parquet_year_path": clean_file_path(dir, "data", "baby_names.parquet"),
        "parquet_state_path": clean_file_path(dir, "data", "baby_names_state.parquet"),
}

generate_file_paths_yaml(
    file_path_dict=file_paths, 
    yaml_output_path=clean_file_path(dir, "utilities", "file_paths.yaml")
)

if aggregate_dataset:
    df_nat = combine_all_txt_files_df(file_paths["dataset_year_path"])
    df_nat.write_parquet(file_paths["parquet_year_path"])

    df_state = combine_all_txt_files_df(file_paths["dataset_state_path"], dataset_type="state")
    df_state.write_parquet(file_paths["parquet_state_path"])
else:
    df_nat = pl.read_parquet(file_paths["parquet_year_path"])
    df_state = pl.read_parquet(file_paths["parquet_state_path"])

