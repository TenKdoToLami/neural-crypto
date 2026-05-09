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
import random

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor

class HDDOptimizedDataset(Dataset):
    """
    Optimized for slow disks (HDDs). 
    Processes assets one by one to minimize disk seeking.
    """
    def __init__(self, raw_path='data/raw/*.csv', processed_dir='data/processed', lookback=100, horizon=16, rebuild_cache=False):
        self.processor = DataProcessor(lookback=lookback, horizon=horizon)
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        raw_files = glob(raw_path)
        self.asset_files = []
        
        print("Verifying/Building cache (Sequential I/O)...")
        for f in tqdm(raw_files, desc="Caching Assets"):
            asset_name = os.path.basename(f).replace('.csv', '.pt')
            processed_path = os.path.join(self.processed_dir, asset_name)
            
            if not os.path.exists(processed_path) or rebuild_cache:
                try:
                    df = pd.read_csv(f)
                    if len(df) < lookback + horizon + 50:
                        continue
                    
                    df = self.processor.add_indicators(df)
                    df = self.processor.create_labels(df)
                    X, y = self.processor.prepare_sequences(df)
                    
                    X = torch.from_numpy(X).float()
                    y = torch.from_numpy(y).float().unsqueeze(1)
                    
                    torch.save({'X': X, 'y': y}, processed_path)
                except Exception as e:
                    print(f"Error caching {f}: {e}")
                    continue
            
            if os.path.exists(processed_path):
                self.asset_files.append(processed_path)
        
        print(f"Ready with {len(self.asset_files)} assets.")

    def get_asset_loader(self, file_path, batch_size):
        """Loads one asset and returns a shuffled DataLoader for its samples."""
        data = torch.load(file_path, weights_only=True)
        ds = torch.utils.data.TensorDataset(data['X'], data['y'])
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Hyperparameters
    BATCH_SIZE = 1024 
    LEARNING_RATE = 2e-4
    EPOCHS = 10 
    
    # 2. Setup Dataset
    dataset = HDDOptimizedDataset()
    
    # 3. Model & Optimizer
    model = NeuralSentinelV1(input_dim=8).to(device)
    
    if hasattr(torch, 'compile'):
        print("Compiling model (this takes a minute)...")
        model = torch.compile(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.BCELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting HDD-Optimized training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_batches = 0
        
        # Shuffle the order of assets each epoch for randomness
        random.shuffle(dataset.asset_files)
        
        pbar = tqdm(dataset.asset_files, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for asset_file in pbar:
            # Load ONE asset into memory (Fast sequential read)
            asset_loader = dataset.get_asset_loader(asset_file, BATCH_SIZE)
            
            asset_loss = 0
            for batch_x, batch_y in asset_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred, _ = model(batch_x)
                    loss = criterion(pred, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                asset_loss += loss.item()
                total_loss += loss.item()
                total_batches += 1
            
            pbar.set_postfix({'asset_loss': f"{asset_loss/len(asset_loader):.4f}"})
            
        print(f"Epoch {epoch+1} summary: Avg Loss: {total_loss/total_batches:.4f}")
        torch.save(model.state_dict(), f"models/sentinel_v1_hdd_optimized.pth")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train()
