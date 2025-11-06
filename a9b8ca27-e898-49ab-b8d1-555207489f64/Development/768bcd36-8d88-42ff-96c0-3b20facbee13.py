import matplotlib.pyplot as plt
import numpy as np

# Additional visualization: Error distribution analysis
fig, _error_axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of prediction errors
_error_axes[0].hist(eval_residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
_error_axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
_error_axes[0].set_xlabel('Prediction Error ($)', fontsize=11, fontweight='bold')
_error_axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
_error_axes[0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
_error_axes[0].legend()
_error_axes[0].grid(True, alpha=0.3)

# Percentage error distribution
eval_pct_errors = ((split_y_test - rf_test_pred) / split_y_test) * 100
_error_axes[1].hist(eval_pct_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
_error_axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
_error_axes[1].set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
_error_axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
_error_axes[1].set_title('Distribution of Percentage Errors', fontsize=12, fontweight='bold')
_error_axes[1].legend()
_error_axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*70)
print("ERROR DISTRIBUTION ANALYSIS")
print("="*70)
print(f"\nAbsolute Error Statistics:")
print(f"  Mean: ${np.mean(np.abs(eval_residuals)):,.2f}")
print(f"  Median: ${np.median(np.abs(eval_residuals)):,.2f}")
print(f"  Std Dev: ${np.std(eval_residuals):,.2f}")
print(f"\nPercentage Error Statistics:")
print(f"  Mean: {np.mean(np.abs(eval_pct_errors)):.2f}%")
print(f"  Median: {np.median(np.abs(eval_pct_errors)):.2f}%")
print("="*70)