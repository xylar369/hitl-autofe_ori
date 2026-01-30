#!/usr/bin/env python
import pandas as pd
import numpy as np
from utils.importance_compute import process_shap_importance
from utils.xgboost_train import train_xgboost_model
import warnings
warnings.filterwarnings('ignore')

# Create a simple test dataset
np.random.seed(42)
n_samples = 100
n_features = 5

X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i+1}' for i in range(n_features)])
X['noise'] = np.random.randn(n_samples) * 0.1
y = (X['f1'] + X['f2'] > 0).astype(int)

# Split into train/val/test
split1 = int(0.6 * n_samples)
split2 = int(0.8 * n_samples)

X_train = X.iloc[:split1]
y_train = y.iloc[:split1]
X_val = X.iloc[split1:split2]
y_val = y.iloc[split1:split2]
X_test = X.iloc[split2:]
y_test = y.iloc[split2:]

print("Dataset created:")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_val columns: {X_val.columns.tolist()}")

try:
    # Train model
    model, _, _, _ = train_xgboost_model(X_train, y_train, X_val, y_val, X_test, y_test, params={})
    print("\nModel trained successfully")
    
    # Test process_shap_importance
    print("\nTesting process_shap_importance...")
    importance_df, useful_features, useless_features, top_features, noise_threshold = process_shap_importance(model, X_val)
    
    print(f"importance_df shape: {importance_df.shape}")
    print(f"importance_df columns: {importance_df.columns.tolist()}")
    print(f"importance_df:\n{importance_df}")
    print(f"\nuseful_features: {useful_features}")
    print(f"useless_features: {useless_features}")
    print(f"top_features: {top_features}")
    print(f"noise_threshold: {noise_threshold}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
