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
    
    # We focus on 32 neurons for the elastic experiment as it was our champion
    hidden_dims = [32]
    
    for hidden_dim in hidden_dims:
        print(f"\n{'='*50}")
        print(f"🚀 STARTING ELASTIC EXPERIMENT: {hidden_dim} Neurons")
        print(f"{'='*50}")
        
        model = NeuralSentinelV1(hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        
        # Add scaler for amp
        scaler = torch.amp.GradScaler('cuda')
        
        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(data)
                    bce_loss = criterion(logits, target)
                    
                    # --- ELASTIC PENALTY (L1) ---
                    l1_penalty = 0
                    for param in model.parameters():
                        l1_penalty += torch.norm(param, 1)
                    
                    loss = bce_loss + (L1_LAMBDA * l1_penalty)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_losses.append(bce_loss.item())
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device).float().unsqueeze(1)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(data)
                        v_loss = criterion(logits, target)
                        val_losses.append(v_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoin with specific naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = f'models/{hidden_dim}N_{timestamp}_sentinel_ELASTIC_E{epoch+1}_L{avg_val_loss:.4f}.pth'
            torch.save(model.state_dict(), save_path)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'models/best_{hidden_dim}N_elastic_model.pth')
                print(f"🏆 New Best Elastic Model! Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    import pandas as pd
    os.makedirs('models', exist_ok=True)
    train_elastic()
