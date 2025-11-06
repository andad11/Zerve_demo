import pandas as pd

# Descriptive statistics for numerical columns
desc_stats = salaries_df.describe()
print("DESCRIPTIVE STATISTICS (Numerical Columns)")
print("=" * 70)
print(desc_stats)
