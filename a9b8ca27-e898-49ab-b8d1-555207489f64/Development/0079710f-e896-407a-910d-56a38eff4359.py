from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Train Gradient Boosting model with optimized hyperparameters
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_model.fit(split_X_train, split_y_train)

# Make predictions
gb_train_pred = gb_model.predict(split_X_train)
gb_test_pred = gb_model.predict(split_X_test)

# Calculate metrics
gb_train_r2 = r2_score(split_y_train, gb_train_pred)
gb_test_r2 = r2_score(split_y_test, gb_test_pred)
gb_train_rmse = np.sqrt(mean_squared_error(split_y_train, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(split_y_test, gb_test_pred))
gb_train_mae = mean_absolute_error(split_y_train, gb_train_pred)
gb_test_mae = mean_absolute_error(split_y_test, gb_test_pred)

print("="*60)
print("GRADIENT BOOSTING MODEL")
print("="*60)
print(f"\nTraining Performance:")
print(f"  R² Score: {gb_train_r2:.4f}")
print(f"  RMSE: ${gb_train_rmse:,.2f}")
print(f"  MAE: ${gb_train_mae:,.2f}")
print(f"\nTest Performance:")
print(f"  R² Score: {gb_test_r2:.4f}")
print(f"  RMSE: ${gb_test_rmse:,.2f}")
print(f"  MAE: ${gb_test_mae:,.2f}")
print("="*60)