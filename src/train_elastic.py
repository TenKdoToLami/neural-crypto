import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import NeuralSentinelV1
from src.data.processor import DataProcessor

class SmartRAMDataset(Dataset):
    def __init__(self, lookback=100, horizon=16, rebuild_cache=False):
        self.lookback = lookback
        self.horizon = horizon
        self.processor = DataProcessor(lookback, horizon)
        self.assets_data = [] # List of (X_tensor, y_tensor)
        self.indices = []     # List of (asset_idx, start_idx)
        
        self.load_data(rebuild_cache)

    def load_data(self, rebuild_cache):
        raw_path = 'data/raw/*.csv'
        from glob import glob
        files = glob(raw_path)
        
        cache_dir = 'data/processed_flat'
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"🧠 Loading assets into Elastic-RAM...")
        for i, f in enumerate(tqdm(files, desc="Processing Assets")):
            asset_name = os.path.basename(f).replace('.csv', '')
            cache_path = os.path.join(cache_dir, f"{asset_name}.pt")
            
            if os.path.exists(cache_path) and not rebuild_cache:
                data = torch.load(cache_path, weights_only=True)
                X_tensor, y_tensor = data['X'], data['y']
            else:
                df = pd.read_csv(f)
                df = self.processor.add_indicators(df)
                df = self.processor.create_labels(df)
                X, y = self.processor.prepare_features(df)
                
                X_tensor = torch.from_numpy(X).float()
                y_tensor = torch.from_numpy(y).float()
                torch.save({'X': X_tensor, 'y': y_tensor}, cache_path)
            
            self.assets_data.append((X_tensor, y_tensor))
            
            # Map valid windows
            num_samples = len(X_tensor) - self.lookback - self.horizon
            for j in range(num_samples):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        asset_idx, start_idx = self.indices[idx]
        X_full, y_full = self.assets_data[asset_idx]
        window = X_full[start_idx : start_idx + self.lookback]
        label = y_full[start_idx + self.lookback]
        return window, label

def train_elastic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧬 Starting ELASTIC BRAIN Training on: {device}")
    
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 2048 
    LEARNING_RATE = 2e-4
    EPOCHS = 10 
    L1_LAMBDA = 1e-5  # The "Elastic Tax" (Reward for being small)
    best_val_loss = float('inf')
    
    # Load Dataset
    dataset = SmartRAMDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = NeuralSentinelV1(input_dim=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_l1_penalty = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch_x)
                
                # 1. Standard Prediction Loss
                base_loss = criterion(logits, batch_y)
                
                # 2. ELASTIC REWARD (L1 Regularization)
                # We calculate the absolute sum of all weights
                l1_penalty = 0
                for param in model.parameters():
                    l1_penalty += torch.norm(param, 1)
                
                # Final Loss = Accuracy + Tax for being big
                loss = base_loss + (L1_LAMBDA * l1_penalty)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_l1_penalty += l1_penalty.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'tax': f"{l1_penalty.item():.1f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Pass
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
        print(f"Epoch {epoch+1} summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Tax Paid: {total_l1_penalty/len(train_loader):.1f}")
        
        # Save Logic
        timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M")
        model_name = f"{timestamp}_sentinel_ELASTIC_E{epoch+1}_L{avg_val_loss:.4f}.pth"
        torch.save(model.state_dict(), os.path.join('models', model_name))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_elastic_model.pth')
            print(f"🏆 New Best Elastic Model! Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    import pandas as pd
    os.makedirs('models', exist_ok=True)
    train_elastic()
