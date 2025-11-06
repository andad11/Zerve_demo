import pandas as pd

# Data types information
print("DATA TYPES")
print("=" * 70)
dtypes_info = salaries_df.dtypes
for col, dtype in dtypes_info.items():
    print(f"{col:25s} : {dtype}")
