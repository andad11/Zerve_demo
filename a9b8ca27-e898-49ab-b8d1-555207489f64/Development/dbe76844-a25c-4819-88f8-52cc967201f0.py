import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a copy for encoding
encoded_df = salaries_df.copy()

# Identify categorical columns (excluding salary fields which are target-related)
cat_cols_to_encode = ['experience_level', 'employment_type', 'job_title', 
                   'employee_residence', 'company_location', 'company_size']

# Encode categorical variables
label_encoders = {}
for _col in cat_cols_to_encode:
    _le = LabelEncoder()
    encoded_df[f'{_col}_encoded'] = _le.fit_transform(encoded_df[_col])
    label_encoders[_col] = _le

print(f"Encoded {len(cat_cols_to_encode)} categorical features")
print(f"New encoded columns: {[f'{_col}_encoded' for _col in cat_cols_to_encode]}")
print(f"Encoded dataframe shape: {encoded_df.shape}")