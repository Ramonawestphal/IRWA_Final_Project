import pandas as pd
import json
import numpy as np

df = pd.read_parquet("data/processed/products_clean.parquet")
example = df.iloc[0].to_dict()

# Convert arrays → lists and safely stringify any other non-serializable objects
for k, v in example.items():
    if isinstance(v, np.ndarray):
        example[k] = v.tolist()

print("### Example of a processed document ###\n")
print(json.dumps(example, indent=4, ensure_ascii=False, default=str))
