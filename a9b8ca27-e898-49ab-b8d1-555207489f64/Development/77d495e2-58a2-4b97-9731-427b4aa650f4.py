import pandas as pd
from scipy import stats
import numpy as np

# Perform ANOVA F-test for categorical features vs salary_in_usd
# This tests if mean salaries differ significantly across categories

cat_features_for_anova = ['experience_level', 'employment_type', 'company_size']

anova_results = []

for _anova_col in cat_features_for_anova:
    # Get groups
    _groups = [group['salary_in_usd'].values for name, group in salaries_df.groupby(_anova_col)]
    
    # Perform one-way ANOVA
    _f_stat, _p_value = stats.f_oneway(*_groups)
    
    anova_results.append({
        'feature': _anova_col,
        'f_statistic': _f_stat,
        'p_value': _p_value,
        'significant': 'Yes' if _p_value < 0.05 else 'No'
    })
    
anova_df = pd.DataFrame(anova_results).sort_values('f_statistic', ascending=False)

print("ANOVA F-Test Results (Categorical Features vs Salary):")
print("="*70)
print(anova_df.to_string(index=False))
print("\nInterpretation:")
print("- Higher F-statistic = stronger relationship with salary")
print("- p-value < 0.05 = statistically significant relationship")