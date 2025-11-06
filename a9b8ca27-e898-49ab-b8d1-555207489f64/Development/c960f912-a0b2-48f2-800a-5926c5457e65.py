import pandas as pd

# Collect all model results
model_results = {
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'Train R²': [linear_train_r2, rf_train_r2, gb_train_r2],
    'Test R²': [linear_test_r2, rf_test_r2, gb_test_r2],
    'Train RMSE': [linear_train_rmse, rf_train_rmse, gb_train_rmse],
    'Test RMSE': [linear_test_rmse, rf_test_rmse, gb_test_rmse],
    'Train MAE': [linear_train_mae, rf_train_mae, gb_train_mae],
    'Test MAE': [linear_test_mae, rf_test_mae, gb_test_mae]
}

comparison_df = pd.DataFrame(model_results)

# Sort by test R² score (best performance)
comparison_df = comparison_df.sort_values('Test R²', ascending=False)

print("="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print("\nRanked by Test R² Score (Higher is Better):\n")
print(comparison_df.to_string(index=False))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_test_r2 = comparison_df.iloc[0]['Test R²']

print("\n" + "="*80)
print(f"BEST PERFORMER: {best_model_name}")
print(f"Test R² Score: {best_test_r2:.4f}")
print("="*80)