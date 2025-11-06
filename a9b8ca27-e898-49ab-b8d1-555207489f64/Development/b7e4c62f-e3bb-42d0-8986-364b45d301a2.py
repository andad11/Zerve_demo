# Export the best performing model for future use
best_trained_model = rf_model
best_model_metrics = {
    'model_name': 'Random Forest',
    'test_r2': rf_test_r2,
    'test_rmse': rf_test_rmse,
    'test_mae': rf_test_mae,
    'train_r2': rf_train_r2,
    'train_rmse': rf_train_rmse,
    'train_mae': rf_train_mae,
    'n_estimators': 200,
    'max_depth': 15,
    'features': list(split_X_train.columns)
}

print("="*70)
print("BEST MODEL READY FOR PREDICTIONS")
print("="*70)
print(f"\nModel Type: Random Forest")
print(f"Training Samples: {len(split_X_train)}")
print(f"Test Samples: {len(split_X_test)}")
print(f"\nTest Performance:")
print(f"  R² Score: {rf_test_r2:.4f}")
print(f"  RMSE: ${rf_test_rmse:,.2f}")
print(f"  MAE: ${rf_test_mae:,.2f}")
print(f"\nFeatures Used ({len(best_model_metrics['features'])}):")
for _feat in best_model_metrics['features']:
    print(f"  - {_feat}")
print("\n✓ Model trained and ready for predictions!")
print("="*70)