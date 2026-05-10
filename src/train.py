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

class SlimRAMDataset(Dataset):
    """
    Fits 100 assets into ~11GB of RAM using float16 storage.
    """
    def __init__(self, raw_path='data/raw/*.csv', processed_dir='data/processed', lookback=100, horizon=16, rebuild_cache=False):
        self.processor = DataProcessor(lookback=lookback, horizon=horizon)
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        raw_files = glob(raw_path)
        all_X = []
        all_y = []
        
        print("🧠 Loading 100 assets into Slim-RAM (float16)...")
        for f in tqdm(raw_files, desc="Compressing Assets"):
            asset_name = os.path.basename(f).replace('.csv', '.pt')
            processed_path = os.path.join(self.processed_dir, asset_name)
            
            if not os.path.exists(processed_path) or rebuild_cache:
                try:
                    df = pd.read_csv(f)
                    if len(df) < lookback + horizon + 50: continue
                    df = self.processor.add_indicators(df)
                    df = self.processor.create_labels(df)
                    X_np, y_np = self.processor.prepare_sequences(df)
                    X = torch.from_numpy(X_np).float()
                    y = torch.from_numpy(y_np).float().unsqueeze(1)
                    torch.save({'X': X, 'y': y}, processed_path)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    continue
            
            # Load and immediately convert to half precision (float16)
            data = torch.load(processed_path, weights_only=True)
            all_X.append(data['X'].half())
            all_y.append(data['y'].half())
        
        print("🔗 Finalizing memory (Concatenating)...")
        self.X = torch.cat(all_X, dim=0)
        self.y = torch.cat(all_y, dim=0)
        
        # Calculate size
        size_gb = (self.X.element_size() * self.X.nelement() + self.y.element_size() * self.y.nelement()) / 1e9
        print(f"✅ Dataset Ready in Slim-RAM: {len(self.X)} samples (~{size_gb:.1f} GB)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert back to float32 on-the-fly for the model
        return self.X[idx].float(), self.y[idx].float()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Hyperparameters
    BATCH_SIZE = 1024 
    LEARNING_RATE = 2e-4
    EPOCHS = 10 
    
    # 2. Load Dataset
    dataset = SlimRAMDataset()
    
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
        
        # Save timestamped model for historical comparison
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        model_name = f"sentinel_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join('models', model_name))
        
        # Also overwrite best_model.pth for the live trader to use immediately
        torch.save(model.state_dict(), f"models/best_model.pth")
        print(f"[*] Model saved as {model_name} and best_model.pth")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train()
