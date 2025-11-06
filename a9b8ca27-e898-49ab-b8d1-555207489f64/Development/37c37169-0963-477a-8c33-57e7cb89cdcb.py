import pandas as pd

# Load the dataset from the file system
salaries_df = pd.read_csv('ds_salaries.csv', index_col=0)

# Display basic information about the dataset structure
print(f"Dataset loaded successfully: {salaries_df.shape[0]} rows, {salaries_df.shape[1]} columns")
print(f"\nColumns: {list(salaries_df.columns)}")
print(f"\nData types:\n{salaries_df.dtypes}")
print(salaries_df.head())