from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

# Train XGBoost-style Gradient Boosting model with optimized hyperparameters
xgb_model = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    random_state=42,
    subsample=0.8,
    max_features='sqrt'
)
xgb_model.fit(split_X_train, split_y_train)

# Make predictions
xgb_train_pred = xgb_model.predict(split_X_train)
xgb_test_pred = xgb_model.predict(split_X_test)

# Calculate metrics
xgb_train_r2 = r2_score(split_y_train, xgb_train_pred)
xgb_test_r2 = r2_score(split_y_test, xgb_test_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(split_y_train, xgb_train_pred))
xgb_test_rmse = np.sqrt(mean_squared_error(split_y_test, xgb_test_pred))
xgb_train_mae = mean_absolute_error(split_y_train, xgb_train_pred)
xgb_test_mae = mean_absolute_error(split_y_test, xgb_test_pred)

# Extract feature importance
xgb_importance = pd.DataFrame({
    'feature': split_X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("="*60)
print("XGBOOST-STYLE GRADIENT BOOSTING MODEL")
print("="*60)
print(f"\nTraining Performance:")
print(f"  R² Score: {xgb_train_r2:.4f}")
print(f"  RMSE: ${xgb_train_rmse:,.2f}")
print(f"  MAE: ${xgb_train_mae:,.2f}")
print(f"\nTest Performance:")
print(f"  R² Score: {xgb_test_r2:.4f}")
print(f"  RMSE: ${xgb_test_rmse:,.2f}")
print(f"  MAE: ${xgb_test_mae:,.2f}")
print("\n" + "-"*60)
print("FEATURE IMPORTANCES (Top 8):")
print("-"*60)
for _idx, _row in xgb_importance.iterrows():
    print(f"  {_row['feature']:30s}: {_row['importance']:.4f}")
print("="*60)