import matplotlib.pyplot as plt
import numpy as np

# Create predicted vs actual visualization for XGBoost
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(split_y_test, xgb_test_pred, alpha=0.6, s=80, color='#2E86AB', edgecolor='white', linewidth=0.5)

# Perfect prediction line (diagonal)
_min_val = min(split_y_test.min(), xgb_test_pred.min())
_max_val = max(split_y_test.max(), xgb_test_pred.max())
ax.plot([_min_val, _max_val], [_min_val, _max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.8)

# Labels and title
ax.set_xlabel('Actual Salary (USD)', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Salary (USD)', fontsize=13, fontweight='bold')
ax.set_title('XGBoost Model: Predicted vs Actual Salary', fontsize=15, fontweight='bold', pad=15)

# Format tick labels as currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Add metrics text box
_metrics_text = f'Test Set Metrics (n={len(split_y_test)}):\n'
_metrics_text += f'R² = {xgb_test_r2:.4f}\n'
_metrics_text += f'MAPE = {xgb_test_mape:.2f}%\n'
_metrics_text += f'RMSE = ${xgb_test_rmse:,.0f}\n'
_metrics_text += f'MAE = ${xgb_test_mae:,.0f}'

ax.text(0.05, 0.95, _metrics_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("✓ XGBoost predicted vs actual visualization created")