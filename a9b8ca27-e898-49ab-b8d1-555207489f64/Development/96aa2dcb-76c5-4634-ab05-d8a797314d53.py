import numpy as np

# Calculate MAPE (Mean Absolute Percentage Error)
# MAPE = mean(|actual - predicted| / |actual|) * 100
def calculate_mape(y_true, y_pred):
    """Calculate MAPE, handling zero values"""
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Avoid division by zero - only calculate for non-zero actuals
    non_zero_mask = y_true_array != 0
    mape_val = np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask])) * 100
    
    return mape_val

# Calculate MAPE for training and test sets
eval_train_mape = calculate_mape(split_y_train, rf_train_pred)
eval_test_mape = calculate_mape(split_y_test, rf_test_pred)

print("="*70)
print("MODEL EVALUATION - MAPE (Mean Absolute Percentage Error)")
print("="*70)
print(f"\nTraining MAPE: {eval_train_mape:.2f}%")
print(f"Test MAPE: {eval_test_mape:.2f}%")
print("\nInterpretation:")
print(f"  On average, predictions are off by {eval_test_mape:.2f}% from actual salaries")
print("="*70)