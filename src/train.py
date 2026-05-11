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

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor

class SmartRAMDataset(Dataset):
    """
    High-Performance Dataset:
    - Stores data FLAT (No window duplication).
    - Uses only ~600MB of RAM for 200 assets.
    - Slices windows on-the-fly in __getitem__.
    """
    def __init__(self, raw_path='data/raw/*.csv', processed_dir='data/processed_flat', lookback=100, horizon=16, rebuild_cache=False):
        self.processor = DataProcessor(lookback=lookback, horizon=horizon)
        self.lookback = lookback
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        raw_files = glob(raw_path)
        self.assets_data = [] # List of (X_tensor, y_tensor)
        self.indices = []     # List of (asset_idx, start_time_idx)
        
        print(f"🧠 Loading {len(raw_files)} assets into Smart-RAM...")
        for asset_idx, f in enumerate(tqdm(raw_files, desc="Processing Assets")):
            asset_name = os.path.basename(f).replace('.csv', '.pt')
            processed_path = os.path.join(self.processed_dir, asset_name)
            
            if not os.path.exists(processed_path) or rebuild_cache:
                try:
                    df = pd.read_csv(f)
                    if len(df) < lookback + horizon + 50: continue
                    df = self.processor.add_indicators(df)
                    df = self.processor.create_labels(df)
                    
                    # Store as FLAT tensors (No windowing yet)
                    X_np, y_np = self.processor.prepare_features(df)
                    X = torch.from_numpy(X_np).float()
                    y = torch.from_numpy(y_np).float().unsqueeze(1)
                    torch.save({'X': X, 'y': y}, processed_path)
                except Exception:
                    continue
            
            try:
                data = torch.load(processed_path, weights_only=True)
                X, y = data['X'], data['y']
                self.assets_data.append((X, y))
                
                # Create valid start indices for this asset
                # We need lookback candles for X, and we need horizon candles ahead for y
                num_samples = len(X) - lookback - horizon
                for i in range(num_samples):
                    self.indices.append((len(self.assets_data)-1, i))
            except Exception:
                continue
        
        print(f"✅ Dataset Ready: {len(self.indices)} samples across {len(self.assets_data)} assets.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        asset_idx, start_idx = self.indices[idx]
        X_full, y_full = self.assets_data[asset_idx]
        
        # Slice window on-the-fly
        window = X_full[start_idx : start_idx + self.lookback]
        label = y_full[start_idx + self.lookback]
        
        return window, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Hyperparameters
    BATCH_SIZE = 1024 
    LEARNING_RATE = 2e-4
    EPOCHS = 10 
    
    # 2. Load Dataset
    dataset = SmartRAMDataset()
    
    if len(dataset) == 0:
        print("[!] No data loaded. Check data/raw directory.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model & Optimizer
    model = NeuralSentinelV1(input_dim=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting High-Performance training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} summary: Avg Loss: {total_loss/len(train_loader):.4f}")
        
        # Save timestamped model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        model_name = f"sentinel_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join('models', model_name))
        
        # Update best_model.pth
        torch.save(model.state_dict(), f"models/best_model.pth")
        print(f"[*] Model saved as {model_name} and best_model.pth")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train()
