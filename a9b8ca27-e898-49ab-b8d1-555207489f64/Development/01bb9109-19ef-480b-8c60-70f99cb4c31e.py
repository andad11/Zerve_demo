import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a copy for encoding
preprocessing_df = salaries_df.copy()

# Remove salary_currency as it's not needed for USD salary prediction
# Keep salary for potential use but the target will be salary_in_usd
preprocessing_df = preprocessing_df.drop(columns=['salary_currency', 'salary'])

# Identify categorical columns to encode
categorical_columns = ['experience_level', 'employment_type', 'job_title', 
                       'employee_residence', 'company_location', 'company_size']

# Store label encoders for potential future use
encoders_dict = {}

# Encode each categorical variable
for cat_col in categorical_columns:
    encoder = LabelEncoder()
    preprocessing_df[f'{cat_col}_encoded'] = encoder.fit_transform(preprocessing_df[cat_col])
    encoders_dict[cat_col] = encoder

print(f"✓ Encoded {len(categorical_columns)} categorical features")
print(f"\nOriginal categorical columns: {categorical_columns}")
print(f"New encoded columns: {[f'{col}_encoded' for col in categorical_columns]}")
print(f"\nPreprocessed dataframe shape: {preprocessing_df.shape}")
print(f"Columns: {list(preprocessing_df.columns)}")
