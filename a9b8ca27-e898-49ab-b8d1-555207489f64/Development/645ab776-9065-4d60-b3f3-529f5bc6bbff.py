from sklearn.model_selection import train_test_split

# Split the dataset into training (80%) and testing (20%) sets
# Using random_state for reproducibility
# No stratification needed as this is regression (continuous target)
split_X_train, split_X_test, split_y_train, split_y_test = train_test_split(
    model_X, 
    model_y, 
    test_size=0.20, 
    random_state=42
)

print("="*70)
print("TRAIN-TEST SPLIT RESULTS")
print("="*70)
print(f"\n✓ Split ratio: 80% train / 20% test")
print(f"✓ Random state: 42 (for reproducibility)")
print(f"\nTraining set:")
print(f"  - X_train shape: {split_X_train.shape}")
print(f"  - y_train shape: {split_y_train.shape}")
print(f"\nTest set:")
print(f"  - X_test shape: {split_X_test.shape}")
print(f"  - y_test shape: {split_y_test.shape}")
print(f"\nTarget variable distribution:")
print(f"  - Train mean: ${split_y_train.mean():,.0f}")
print(f"  - Test mean: ${split_y_test.mean():,.0f}")
print(f"  - Overall mean: ${model_y.mean():,.0f}")
print("="*70)