import matplotlib.pyplot as plt
import numpy as np

# Create visualization comparing predicted vs actual salaries
fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: Predicted vs Actual
_axes[0].scatter(split_y_test, rf_test_pred, alpha=0.6, edgecolors='k', s=50)
_axes[0].plot([split_y_test.min(), split_y_test.max()], 
              [split_y_test.min(), split_y_test.max()], 
              'r--', lw=2, label='Perfect Prediction')
_axes[0].set_xlabel('Actual Salary ($)', fontsize=11, fontweight='bold')
_axes[0].set_ylabel('Predicted Salary ($)', fontsize=11, fontweight='bold')
_axes[0].set_title('Predicted vs Actual Salaries (Test Set)', fontsize=12, fontweight='bold')
_axes[0].legend()
_axes[0].grid(True, alpha=0.3)

# Residual plot: Prediction errors
eval_residuals = split_y_test - rf_test_pred
_axes[1].scatter(rf_test_pred, eval_residuals, alpha=0.6, edgecolors='k', s=50)
_axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
_axes[1].set_xlabel('Predicted Salary ($)', fontsize=11, fontweight='bold')
_axes[1].set_ylabel('Residual (Actual - Predicted) ($)', fontsize=11, fontweight='bold')
_axes[1].set_title('Residual Plot: Prediction Errors', fontsize=12, fontweight='bold')
_axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*70)
print("VISUALIZATION: Predicted vs Actual Salaries")
print("="*70)
print(f"\nTest Set: {len(split_y_test)} predictions")
print(f"Mean Absolute Error: ${rf_test_mae:,.2f}")
print(f"MAPE: {eval_test_mape:.2f}%")
print("="*70)