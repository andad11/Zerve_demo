import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prepare features for Random Forest
feature_cols = ['work_year', 'remote_ratio', 
                'experience_level_encoded', 'employment_type_encoded', 
                'job_title_encoded', 'employee_residence_encoded', 
                'company_location_encoded', 'company_size_encoded']

X = encoded_df[feature_cols]
y = encoded_df['salary_in_usd']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Calculate R² score
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print("Random Forest Feature Importance:")
print(rf_importance.to_string(index=False))
print(f"\nModel R² - Train: {train_score:.3f}, Test: {test_score:.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(rf_importance['feature'], rf_importance['importance'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest Feature Importance for Salary Prediction', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()