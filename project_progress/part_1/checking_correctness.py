import pandas as pd

# load your cleaned dataset and the validation file
df_clean = pd.read_parquet("data/processed/products_clean.parquet")
df_val = pd.read_csv("data/raw/validation_labels.csv")

print("Cleaned:", df_clean.shape)
print("Validation:", df_val.shape)

# Check if all validation PIDs exist in your cleaned data
missing = set(df_val['pid']) - set(df_clean['pid'])
print(f"Missing pids in cleaned file: {len(missing)}")

# Spot-check a few rows for consistency
sample = df_clean[df_clean['pid'].isin(df_val['pid'].head(5))]
print(sample[['pid','brand','category','selling_price','average_rating']])
