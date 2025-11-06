from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(split_X_train, split_y_train)

# Make predictions
linear_train_pred = linear_model.predict(split_X_train)
linear_test_pred = linear_model.predict(split_X_test)

# Calculate metrics
linear_train_r2 = r2_score(split_y_train, linear_train_pred)
linear_test_r2 = r2_score(split_y_test, linear_test_pred)
linear_train_rmse = np.sqrt(mean_squared_error(split_y_train, linear_train_pred))
linear_test_rmse = np.sqrt(mean_squared_error(split_y_test, linear_test_pred))
linear_train_mae = mean_absolute_error(split_y_train, linear_train_pred)
linear_test_mae = mean_absolute_error(split_y_test, linear_test_pred)

print("="*60)
print("LINEAR REGRESSION MODEL")
print("="*60)
print(f"\nTraining Performance:")
print(f"  R² Score: {linear_train_r2:.4f}")
print(f"  RMSE: ${linear_train_rmse:,.2f}")
print(f"  MAE: ${linear_train_mae:,.2f}")
print(f"\nTest Performance:")
print(f"  R² Score: {linear_test_r2:.4f}")
print(f"  RMSE: ${linear_test_rmse:,.2f}")
print(f"  MAE: ${linear_test_mae:,.2f}")
print("="*60)