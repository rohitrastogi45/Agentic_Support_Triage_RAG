import pandas as pd
import os

# The big files to sample (excluding brand mappings)
files_to_sample = [
    "review_support_pairs.csv",
    "support_pairs.csv",
    "useful_support_pairs.csv"
]

base_path = "data/processed/"

for file_name in files_to_sample:
    file_path = os.path.join(base_path, file_name)
    sample_path = os.path.join(base_path, f"sample_{file_name}")
    
    print(f"Creating sample for {file_name}...")
    try:
        # Read only the first 100 rows to keep it under a few kilobytes
        df = pd.read_csv(file_path, nrows=100)
        df.to_csv(sample_path, index=False)
        print(f"Success: Saved to {sample_path}")
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")