import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Analyze categorical variables and their relationship with salary
cat_features_for_analysis = ['experience_level', 'employment_type', 'company_size', 
                       'employee_residence', 'company_location']

categorical_stats = []

for _cat_col in cat_features_for_analysis:
    # Calculate mean salary and count for each category
    grouped = salaries_df.groupby(_cat_col)['salary_in_usd'].agg(['mean', 'median', 'std', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    # Calculate coefficient of variation (std/mean) to measure salary spread
    _cv = grouped['std'] / grouped['mean']
    
    # Store summary stats
    categorical_stats.append({
        'feature': _cat_col,
        'unique_values': salaries_df[_cat_col].nunique(),
        'salary_range': grouped['mean'].max() - grouped['mean'].min(),
        'max_mean_salary': grouped['mean'].max(),
        'min_mean_salary': grouped['mean'].min(),
        'cv_mean': _cv.mean()
    })
    
    print(f"\n{_cat_col.upper()} - Salary Analysis:")
    print(grouped.head(10).to_string())

# Summary DataFrame
cat_summary_df = pd.DataFrame(categorical_stats).sort_values('salary_range', ascending=False)

print("\n" + "="*70)
print("CATEGORICAL FEATURE SUMMARY (sorted by salary range):")
print("="*70)
print(cat_summary_df.to_string(index=False))

# Visualization: Top categorical features by salary range
plt.figure(figsize=(10, 6))
plt.barh(cat_summary_df['feature'], cat_summary_df['salary_range'])
plt.xlabel('Salary Range ($)', fontsize=12)
plt.title('Salary Range by Categorical Feature', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()