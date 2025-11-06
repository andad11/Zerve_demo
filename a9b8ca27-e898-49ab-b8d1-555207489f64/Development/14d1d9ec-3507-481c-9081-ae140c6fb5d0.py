import pandas as pd

# Select only the encoded features and numerical features for the model
model_features = [
    'work_year',
    'remote_ratio',
    'experience_level_encoded',
    'employment_type_encoded', 
    'job_title_encoded',
    'employee_residence_encoded',
    'company_location_encoded',
    'company_size_encoded'
]

# Create clean feature matrix X and target y
model_X = preprocessing_df[model_features].copy()
model_y = preprocessing_df['salary_in_usd'].copy()

print("✓ Feature matrix created successfully")
print(f"\nFeature matrix shape: {model_X.shape}")
print(f"Target variable shape: {model_y.shape}")
print(f"\nFeatures included:")
for feature in model_features:
    print(f"  - {feature}")
print(f"\nTarget variable: salary_in_usd")
print(f"Target range: ${model_y.min():,.0f} - ${model_y.max():,.0f}")
print(f"Target mean: ${model_y.mean():,.0f}")
