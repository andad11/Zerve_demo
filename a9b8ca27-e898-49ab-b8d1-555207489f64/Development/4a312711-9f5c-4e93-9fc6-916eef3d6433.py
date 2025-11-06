import pandas as pd

# Check for missing values
missing_check = salaries_df.isnull().sum()
total_missing = missing_check.sum()

print("Missing Values Check:")
print(f"Total missing values: {total_missing}")

if total_missing > 0:
    print("\nColumns with missing values:")
    print(missing_check[missing_check > 0])
else:
    print("✓ No missing values found in the dataset")
