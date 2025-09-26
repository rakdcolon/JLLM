import pandas as pd

# Read parquet files
test_df = pd.read_parquet("data/test.parquet")
train_df = pd.read_parquet("data/train.parquet")
validation_df = pd.read_parquet("data/validation.parquet")

# Save as plain text (space-separated or tab-separated)
test_df.to_csv("test.txt", sep=" ", index=False, header=False)
train_df.to_csv("train.txt", sep=" ", index=False, header=False)
validation_df.to_csv("validation.txt", sep=" ", index=False, header=False)