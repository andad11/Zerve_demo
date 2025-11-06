import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation analysis for numerical features with salary_in_usd
numerical_cols = ['work_year', 'salary', 'salary_in_usd', 'remote_ratio']
corr_data = salaries_df[numerical_cols].corr()

# Extract correlations with salary_in_usd
salary_correlations = corr_data['salary_in_usd'].drop('salary_in_usd').sort_values(ascending=False)

print("Correlation with salary_in_usd:")
print(salary_correlations)

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.3f', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()