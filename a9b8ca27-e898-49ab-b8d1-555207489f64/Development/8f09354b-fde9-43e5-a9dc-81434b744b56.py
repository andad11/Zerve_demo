import pandas as pd

# Unique value counts for each column
print("UNIQUE VALUE COUNTS")
print("=" * 70)
for col in salaries_df.columns:
    unique_cnt = salaries_df[col].nunique()
    print(f"{col:25s} : {unique_cnt:4d} unique values")
