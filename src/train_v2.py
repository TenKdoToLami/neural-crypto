import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier_v2 import NeuralSentinelV2
from src.data.processor_v2 import DataProcessorV2

class SmartRAMDatasetV2(Dataset):
    """
    Dataset for V2:
    - Includes VPT feature.
    - Uses flat storage to save RAM.
    """
    def __init__(self, raw_path='data/raw/*.csv', processed_dir='data/processed_v2', lookback=100, horizon=16, rebuild_cache=False):
        self.processor = DataProcessorV2(lookback=lookback, horizon=horizon)
        self.lookback = lookback
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        raw_files = glob(raw_path)
        self.assets_data = [] 
        self.indices = []     
        
        print(f"🧠 Loading {len(raw_files)} assets into Smart-RAM (V2)...")
        for asset_idx, f in enumerate(tqdm(raw_files, desc="Processing V2 Assets")):
            asset_name = os.path.basename(f).replace('.csv', '_v2.pt')
            processed_path = os.path.join(self.processed_dir, asset_name)
            
            if not os.path.exists(processed_path) or rebuild_cache:
                try:
                    df = pd.read_csv(f)
                    if len(df) < lookback + horizon + 50: continue
                    df = self.processor.add_indicators(df)
                    df = self.processor.create_labels(df)
                    
                    X_np, y_np = self.processor.prepare_features(df)
                    X = torch.from_numpy(X_np).float()
                    y = torch.from_numpy(y_np).float().unsqueeze(1)
                    torch.save({'X': X, 'y': y}, processed_path)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    continue
            
            try:
                data = torch.load(processed_path, weights_only=True)
                X, y = data['X'], data['y']
                self.assets_data.append((X, y))
                
                num_samples = len(X) - lookback - horizon
                for i in range(num_samples):
                    self.indices.append((len(self.assets_data)-1, i))
            except Exception:
                continue
        
        print(f"✅ V2 Dataset Ready: {len(self.indices)} samples across {len(self.assets_data)} assets.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        asset_idx, start_idx = self.indices[idx]
        X_full, y_full = self.assets_data[asset_idx]
        window = X_full[start_idx : start_idx + self.lookback]
        label = y_full[start_idx + self.lookback]
        return window, label

def train_v2(hidden_dim=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "🚀"*20)
    print(f"🧬 V2 EXPERIMENT: {hidden_dim} NEURONS")
    print(f"🧬 Training on: {device}")
    print("🚀"*20 + "\n")
    
    # 1. Hyperparameters
    BATCH_SIZE = 2048 
    LEARNING_RATE = 1e-4 # Slightly lower for V2 stability
    EPOCHS = 60 
    SUBSAMPLE_PCT = 0.30 # Slightly more data for the more complex V2 model
    NOISE_LEVEL = 0.015   # Increased noise for better generalization
    
    # 2. Load Dataset
    dataset = SmartRAMDatasetV2()
    
    if len(dataset) == 0:
        print("[!] No data loaded. Check data/raw directory.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. Model & Optimizer
    model = NeuralSentinelV2(input_dim=9, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    print(f"Starting V2 training (Subsampling: {SUBSAMPLE_PCT:.0%}, Features: 9)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Subsampling
        indices = torch.randperm(len(train_ds))[:int(len(train_ds) * SUBSAMPLE_PCT)]
        epoch_ds = torch.utils.data.Subset(train_ds, indices)
        epoch_loader = DataLoader(epoch_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
        
        pbar = tqdm(epoch_loader, desc=f"[V2-{hidden_dim}N] Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Noise injection
            if model.training:
                noise = torch.randn_like(batch_x) * NOISE_LEVEL
                batch_x = batch_x + noise
                
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(epoch_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(batch_x)
                    loss = criterion(logits, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"[V2-{hidden_dim}N] Epoch {epoch+1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")
        
        # Save Logic
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"V2_{hidden_dim}N_{timestamp}_E{epoch+1}_L{avg_val_loss:.4f}.pth"
        torch.save(model.state_dict(), os.path.join('models', model_name))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/best_V2_{hidden_dim}N_model.pth')
            print(f"🏆 New Best V2-{hidden_dim}N! Val Loss: {avg_val_loss:.4f}")

    del model
    del optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    # EXPERIMENT_NEURONS = [16, 20, 24, 28, 32, 64]
    EXPERIMENT_NEURONS = [24, 28, 32, 64] # Skipping 16 as it finished
    for n in EXPERIMENT_NEURONS:
        train_v2(hidden_dim=n)
    print("\n✅ V2 Training Complete!")
