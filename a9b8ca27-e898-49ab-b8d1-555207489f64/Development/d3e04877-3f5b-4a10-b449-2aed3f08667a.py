import pandas as pd

# Comprehensive evaluation report with all requested metrics
eval_metrics_summary = {
    'Metric': ['MAPE (%)', 'MAE ($)', 'RMSE ($)', 'R² Score'],
    'Training Set': [
        eval_train_mape,
        rf_train_mae,
        rf_train_rmse,
        rf_train_r2
    ],
    'Test Set': [
        eval_test_mape,
        rf_test_mae,
        rf_test_rmse,
        rf_test_r2
    ]
}

eval_metrics_df = pd.DataFrame(eval_metrics_summary)

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION REPORT")
print("="*80)
print(f"\nModel: Random Forest Regressor")
print(f"Training Samples: {len(split_X_train)} | Test Samples: {len(split_X_test)}")
print(f"\n{eval_metrics_df.to_string(index=False)}")
print("\n" + "="*80)
print("PERFORMANCE INTERPRETATION:")
print("="*80)
print(f"\n✓ Primary Metric - MAPE: {eval_test_mape:.2f}%")
print(f"  → Model predictions deviate by ~{eval_test_mape:.2f}% on average from actual salaries")
print(f"\n✓ MAE: ${rf_test_mae:,.2f}")
print(f"  → Average prediction error of ${rf_test_mae:,.0f}")
print(f"\n✓ RMSE: ${rf_test_rmse:,.2f}")
print(f"  → Root mean squared error of ${rf_test_rmse:,.0f} (penalizes larger errors)")
print(f"\n✓ R² Score: {rf_test_r2:.4f}")
print(f"  → Model explains {rf_test_r2*100:.2f}% of salary variance")
print("="*80)