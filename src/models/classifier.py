import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions to capture multi-scale temporal patterns.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn(self.conv(x)))
        out += self.shortcut(residual)
        return out

class NeuralSentinelV1(nn.Module):
    """
    Optimized architecture for RTX 4070.
    Hybrid CNN-Transformer-GRU with Dual Output heads.
    """
    def __init__(self, input_dim=8, hidden_dim=256, num_heads=8, num_layers=2):
        super(NeuralSentinelV1, self).__init__()
        
        # 1. Feature Extraction (Dilated CNN)
        self.feat_extract = nn.Sequential(
            DilatedResidualBlock(input_dim, 128, dilation=1),
            DilatedResidualBlock(128, 128, dilation=2),
            DilatedResidualBlock(128, hidden_dim, dilation=4)
        )
        
        # 2. Temporal Attention (Transformer)
        # Using Batch First for easier data handling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Sequential Refinement (GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # 4. Dual Heads
        # Head 1: Rally Probability (Classification)
        self.rally_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Head 2: Volatility/Magnitude Estimate (Regression)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # CNN expects [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.feat_extract(x)
        
        # Back to [batch, seq_len, hidden_dim] for Transformer
        x = x.transpose(1, 2)
        x = self.transformer(x)
        
        # GRU refinement
        x, _ = self.gru(x)
        
        # Use last time step for heads
        last_step = x[:, -1, :]
        
        rally_logits = self.rally_head(last_step)
        vol_est = self.vol_head(last_step)
        
        return rally_logits, vol_est

if __name__ == "__main__":
    # Test model shape and check for CUDA optimization compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentinelV1().to(device)
    
    # Batch size of 256 to simulate RTX 4070 load
    dummy_input = torch.randn(256, 100, 8).to(device)
    
    # Benchmark forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16): # Simulate BF16
        rally, vol = model(dummy_input)
    
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Rally Output shape: {rally.shape}")
    print(f"Vol Output shape: {vol.shape}")
    print("Model initialized and verified for Mixed Precision (BF16).")
