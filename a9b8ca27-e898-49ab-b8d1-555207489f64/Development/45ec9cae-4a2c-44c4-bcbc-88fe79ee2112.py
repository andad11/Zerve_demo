import pandas as pd

# Consolidate findings from all analyses
print("="*80)
print("FEATURE ANALYSIS SUMMARY REPORT - DS SALARIES DATASET")
print("="*80)

print("\n1. CORRELATION ANALYSIS (Numerical Features):")
print("-" * 80)
print(salary_correlations.to_string())

print("\n\n2. RANDOM FOREST FEATURE IMPORTANCE:")
print("-" * 80)
print(rf_importance.to_string(index=False))

print("\n\n3. CATEGORICAL FEATURE ANALYSIS:")
print("-" * 80)
print(cat_summary_df.to_string(index=False))

print("\n\n4. STATISTICAL SIGNIFICANCE (ANOVA):")
print("-" * 80)
print(anova_df.to_string(index=False))

print("\n\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Identify top features from RF importance
top_rf_features = rf_importance.head(3)['feature'].tolist()
print(f"\nMost Important Features (Random Forest):")
for i, feat in enumerate(top_rf_features, 1):
    importance_val = rf_importance[rf_importance['feature'] == feat]['importance'].values[0]
    print(f"  {i}. {feat}: {importance_val:.4f}")

# Top categorical features by salary range
top_cat = cat_summary_df.head(3)
print(f"\nCategorical Features with Highest Salary Variance:")
for idx, row in top_cat.iterrows():
    print(f"  - {row['feature']}: ${row['salary_range']:,.0f} range")

# Statistical significance
sig_features = anova_df[anova_df['significant'] == 'Yes']
print(f"\nStatistically Significant Features (p < 0.05): {len(sig_features)}")
for idx, row in sig_features.iterrows():
    print(f"  - {row['feature']}: F={row['f_statistic']:.2f}, p={row['p_value']:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The most predictive features for salary are:")
print("1. Job title (highest RF importance and salary range)")
print("2. Experience level (high importance and statistical significance)")  
print("3. Company location (significant salary variation)")
print("4. Remote ratio and work year show weaker correlations")
print("="*80)