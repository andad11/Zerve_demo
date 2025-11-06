import pandas as pd

# Missing values analysis
print("MISSING VALUES ANALYSIS")
print("=" * 70)
missing_count = salaries_df.isnull().sum()
missing_pct = (salaries_df.isnull().sum() / len(salaries_df)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %': missing_pct
})

print(missing_summary)
print(f"\nTotal missing values: {missing_count.sum()}")
