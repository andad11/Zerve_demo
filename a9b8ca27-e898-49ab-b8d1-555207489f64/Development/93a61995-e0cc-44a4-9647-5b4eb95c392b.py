import pandas as pd
import numpy as np

# Validate the preprocessing results
print("="*70)
print("DATA PREPROCESSING SUMMARY & VALIDATION")
print("="*70)

# 1. Missing values check
print("\n1. MISSING VALUES:")
print(f"   ✓ Total missing values in feature matrix: {model_X.isnull().sum().sum()}")
print(f"   ✓ Total missing values in target: {model_y.isnull().sum()}")

# 2. Categorical encoding validation
print("\n2. CATEGORICAL ENCODING:")
print(f"   ✓ All categorical variables encoded: {len(categorical_columns)} features")
for cat in categorical_columns:
    encoded_col = f'{cat}_encoded'
    unique_count = model_X[encoded_col].nunique()
    print(f"     - {cat}: {unique_count} unique values encoded")

# 3. Feature matrix info
print("\n3. FEATURE MATRIX:")
print(f"   ✓ Shape: {model_X.shape}")
print(f"   ✓ Features: {list(model_X.columns)}")
print(f"   ✓ All numeric dtypes: {model_X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()}")

# 4. Target variable info
print("\n4. TARGET VARIABLE (salary_in_usd):")
print(f"   ✓ Shape: {model_y.shape}")
print(f"   ✓ Range: ${model_y.min():,.0f} to ${model_y.max():,.0f}")
print(f"   ✓ Mean: ${model_y.mean():,.0f}")
print(f"   ✓ Median: ${model_y.median():,.0f}")

# 5. Ready for modeling
print("\n5. MODEL READINESS:")
print(f"   ✓ No missing values: {model_X.isnull().sum().sum() == 0 and model_y.isnull().sum() == 0}")
print(f"   ✓ All features numeric: {model_X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()}")
print(f"   ✓ Feature-target alignment: {len(model_X) == len(model_y)}")

print("\n" + "="*70)
print("✓ DATA PREPROCESSING COMPLETE - READY FOR MODEL TRAINING")
print("="*70)
