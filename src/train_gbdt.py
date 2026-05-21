import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor_gbdt import DataProcessorGBDT

def get_approved_assets():
    approved_path = 'data/approved_assets.txt'
    if not os.path.exists(approved_path):
        print(f"[!] Warning: {approved_path} not found. Processing all raw assets.")
        return None
    
    approved = []
    with open(approved_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle formats like BTC/USDC, SOL/USDC -> BTC_USDC
                pair = line.replace('/', '_')
                approved.append(pair)
    return approved

def main():
    print("\n" + "="*30)
    print("GBDT MODEL TRAINING PIPELINE")
    print("="*30 + "\n")

    # 1. Configuration
    DATA_DIR = 'data/raw'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    processor = DataProcessorGBDT(lookback=100, horizon=16)
    approved_pairs = get_approved_assets()

    raw_files = glob(os.path.join(DATA_DIR, "*.csv"))
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    total_raw_rows = 0
    total_processed_rows = 0

    print("[*] Ingesting and processing training data (2024-Present)...")
    for f in tqdm(raw_files, desc="Assets"):
        filename = os.path.basename(f)
        symbol = filename.replace('_15m.csv', '')
        
        # Check if asset is approved (e.g., BTC_USDC)
        if approved_pairs is not None:
            # We want to match asset base names (like BTC_USDC)
            # In data/raw, files might be named like BTC_USDC_15m.csv or BTC_USDT_15m.csv.
            # We match approved pairs
            if symbol not in approved_pairs and symbol.replace('_USDT', '_USDC') not in approved_pairs:
                continue

        try:
            df = pd.read_csv(f)
            if len(df) < 500:
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            total_raw_rows += len(df)

            # Filter for training data starting from 2024
            df_train_period = df[df['timestamp'] >= '2024-01-01'].copy()
            if len(df_train_period) < 200:
                continue

            # Process indicators & labels
            df_proc = processor.add_indicators(df_train_period)
            df_proc = processor.create_labels(df_proc, threshold=0.015)
            
            X, y, feature_names = processor.prepare_features(df_proc)
            if len(X) < 100:
                continue
                
            total_processed_rows += len(X)

            # Chronological split (80% Train, 20% Val) per asset to avoid lookahead leakage
            split_idx = int(len(X) * 0.8)
            X_train_list.append(X[:split_idx])
            y_train_list.append(y[:split_idx])
            X_val_list.append(X[split_idx:])
            y_val_list.append(y[split_idx:])
            
        except Exception as e:
            print(f"Error processing asset {filename}: {e}")
            continue

    if not X_train_list:
        print("[!] No training samples found. Please run fetcher.py first or check data/raw.")
        return

    # Concatenate all assets' data
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    print(f"\n[*] Dataset Summary:")
    print(f"   Total Raw Rows:        {total_raw_rows:,}")
    print(f"   Processed Rows (2024+): {total_processed_rows:,}")
    print(f"   Training Samples:      {X_train.shape[0]:,} (Bullish: {np.sum(y_train == 1):,}, Bearish/Sideways: {np.sum(y_train == 0):,})")
    print(f"   Validation Samples:    {X_val.shape[0]:,} (Bullish: {np.sum(y_val == 1):,}, Bearish/Sideways: {np.sum(y_val == 0):,})")
    print(f"   Features Count:        {X_train.shape[1]}")

    # 2. Train Model
    print("\n[*] Fitting HistGradientBoostingClassifier...")
    # class_weight='balanced' is vital due to class imbalance of bullish swings.
    # We restrict model capacity strictly to focus on general patterns rather than memorization.
    model = HistGradientBoostingClassifier(
        max_iter=80,             # Fewer trees to limit noise memorization
        learning_rate=0.03,      # Slower learning for smoother gradient steps
        max_depth=3,             # Shallow depth (max 8 leaves) to enforce broad simple rules
        min_samples_leaf=200,    # Large cohort constraint to suppress individual candle anomalies
        l2_regularization=15.0,  # Strict weight decay to stabilize predictions
        class_weight='balanced',
        random_state=42,
        verbose=0
    )
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_duration = datetime.now() - start_time
    print(f"[+] Training completed in {training_duration.total_seconds():.2f}s.")

    # 3. Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_prob)
    val_auc = roc_auc_score(y_val, y_val_prob)

    print("\n" + "="*45)
    print("PERFORMANCE REPORT")
    print("="*45)
    print(f"Training ROC-AUC:   {train_auc:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Not Bullish", "Bullish"]))
    
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    print("="*45)

    # 4. Save Model & Metadata
    model_filename = "sentinel_gbdt.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    # Save model alongside feature names for future inference safety
    model_artifact = {
        'model': model,
        'feature_names': feature_names,
        'trained_at': datetime.now().isoformat(),
        'val_auc': val_auc
    }
    
    joblib.dump(model_artifact, model_path)
    print(f"\n[+] Model saved successfully to: {model_path}")

    # Update best_model alias
    best_path = os.path.join(MODEL_DIR, "best_gbdt_model.joblib")
    joblib.dump(model_artifact, best_path)
    print(f"[+] Best model alias updated: {best_path}")

if __name__ == "__main__":
    main()
