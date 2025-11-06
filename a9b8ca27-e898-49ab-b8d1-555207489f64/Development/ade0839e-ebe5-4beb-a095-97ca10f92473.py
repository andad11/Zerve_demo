import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# Calculate MAPE for XGBoost model
xgb_test_mape = mean_absolute_percentage_error(split_y_test, xgb_test_pred) * 100
xgb_train_mape = mean_absolute_percentage_error(split_y_train, xgb_train_pred) * 100

# Display comprehensive XGBoost metrics
print("=" * 70)
print("XGBOOST MODEL - COMPREHENSIVE EVALUATION METRICS")
print("=" * 70)
print(f"\n📊 TEST SET PERFORMANCE (n={len(split_y_test)}):")
print("-" * 70)
print(f"  • MAPE (Mean Absolute Percentage Error): {xgb_test_mape:.2f}%")
print(f"  • MAE (Mean Absolute Error):             ${xgb_test_mae:,.2f}")
print(f"  • RMSE (Root Mean Squared Error):        ${xgb_test_rmse:,.2f}")
print(f"  • R² Score:                               {xgb_test_r2:.4f}")

print(f"\n📈 TRAINING SET PERFORMANCE (n={len(split_y_train)}):")
print("-" * 70)
print(f"  • MAPE:  {xgb_train_mape:.2f}%")
print(f"  • MAE:   ${xgb_train_mae:,.2f}")
print(f"  • RMSE:  ${xgb_train_rmse:,.2f}")
print(f"  • R²:    {xgb_train_r2:.4f}")

print("\n" + "=" * 70)
print(f"✓ XGBoost evaluation complete")
print("=" * 70)