import pandas as pd

# Need to get Random Forest predictions to compare with XGBoost
# First, generate RF predictions using the existing rf_model from feature.summary_report
rf_test_pred = rf_model.predict(X_test)
rf_train_pred = rf_model.predict(X_train)

# Calculate RF metrics for comparison
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_mape = mean_absolute_percentage_error(y_test, rf_test_pred) * 100

rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_mape = mean_absolute_percentage_error(y_train, rf_train_pred) * 100

# Create comparison table
comparison_data = {
    'Model': ['XGBoost', 'Random Forest'],
    'Test_R2': [xgb_test_r2, rf_test_r2],
    'Test_MAPE_%': [xgb_test_mape, rf_test_mape],
    'Test_MAE_$': [xgb_test_mae, rf_test_mae],
    'Test_RMSE_$': [xgb_test_rmse, rf_test_rmse],
    'Train_R2': [xgb_train_r2, rf_train_r2],
    'Train_MAPE_%': [xgb_train_mape, rf_train_mape],
    'Train_MAE_$': [xgb_train_mae, rf_train_mae],
    'Train_RMSE_$': [xgb_train_rmse, rf_train_rmse]
}

comparison_df = pd.DataFrame(comparison_data)

print("=" * 85)
print("XGBOOST vs RANDOM FOREST - COMPREHENSIVE PERFORMANCE COMPARISON")
print("=" * 85)
print("\n📊 TEST SET COMPARISON:")
print("-" * 85)
print(f"{'Metric':<20} {'XGBoost':<20} {'Random Forest':<20} {'Winner':<15}")
print("-" * 85)
print(f"{'R² Score':<20} {xgb_test_r2:<20.4f} {rf_test_r2:<20.4f} {'XGB' if xgb_test_r2 > rf_test_r2 else 'RF':<15}")
print(f"{'MAPE (%)':<20} {xgb_test_mape:<20.2f} {rf_test_mape:<20.2f} {'XGB' if xgb_test_mape < rf_test_mape else 'RF':<15}")
print(f"{'MAE ($)':<20} {xgb_test_mae:<20,.0f} {rf_test_mae:<20,.0f} {'XGB' if xgb_test_mae < rf_test_mae else 'RF':<15}")
print(f"{'RMSE ($)':<20} {xgb_test_rmse:<20,.0f} {rf_test_rmse:<20,.0f} {'XGB' if xgb_test_rmse < rf_test_rmse else 'RF':<15}")

print("\n📈 TRAINING SET COMPARISON:")
print("-" * 85)
print(f"{'Metric':<20} {'XGBoost':<20} {'Random Forest':<20}")
print("-" * 85)
print(f"{'R² Score':<20} {xgb_train_r2:<20.4f} {rf_train_r2:<20.4f}")
print(f"{'MAPE (%)':<20} {xgb_train_mape:<20.2f} {rf_train_mape:<20.2f}")
print(f"{'MAE ($)':<20} {xgb_train_mae:<20,.0f} {rf_train_mae:<20,.0f}")
print(f"{'RMSE ($)':<20} {xgb_train_rmse:<20,.0f} {rf_train_rmse:<20,.0f}")

print("\n" + "=" * 85)
print("🏆 OVERALL ASSESSMENT:")
print("=" * 85)

# Determine overall winner
xgb_wins = sum([
    xgb_test_r2 > rf_test_r2,
    xgb_test_mape < rf_test_mape,
    xgb_test_mae < rf_test_mae,
    xgb_test_rmse < rf_test_rmse
])

print(f"XGBoost wins on {xgb_wins}/4 test metrics")
print(f"Random Forest wins on {4-xgb_wins}/4 test metrics")

if xgb_wins > 2:
    print("\n✓ XGBoost shows better overall performance on the test set")
elif xgb_wins < 2:
    print("\n✓ Random Forest shows better overall performance on the test set")
else:
    print("\n✓ Models show similar overall performance on the test set")

print("=" * 85)