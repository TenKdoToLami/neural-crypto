import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    """
    Squeeze-and-Excitation style block for feature importance.
    Helps the model focus on the most relevant indicators (e.g. VPT vs RSI)
    depending on the current market context.
    """
    def __init__(self, channels):
        super(FeatureAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(channels // 2, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, features, seq_len]
        b, c, s = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DilatedResidualBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlockV2, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.attn = FeatureAttention(out_channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn(self.conv(x)))
        out = self.attn(out)
        out += self.shortcut(residual)
        return out

class NeuralSentinelV2(nn.Module):
    """
    V2 Architecture: Enhanced for RTX 4070.
    - 9 Input Features (includes VPT)
    - Feature Attention Blocks
    - Multi-Head Temporal Attention
    - Dual GRU Refinement
    """
    def __init__(self, input_dim=9, hidden_dim=256, num_heads=4, num_layers=3):
        super(NeuralSentinelV2, self).__init__()
        
        # 1. Spatial-Temporal Feature Extraction
        self.feat_extract = nn.Sequential(
            DilatedResidualBlockV2(input_dim, 128, dilation=1),
            DilatedResidualBlockV2(128, 128, dilation=2),
            DilatedResidualBlockV2(128, hidden_dim, dilation=4)
        )
        
        # 2. Global Context (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Sequential Memory (GRU)
        # Using 2 layers of GRU for better long-term dependency capture
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=0.1)
        
        # 4. Heads
        self.rally_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2) # [batch, features, seq_len]
        x = self.feat_extract(x)
        
        x = x.transpose(1, 2) # [batch, seq_len, hidden_dim]
        x = self.transformer(x)
        
        x, _ = self.gru(x) # [batch, seq_len, hidden_dim]
        
        # Weighted average of temporal states instead of just the last step
        # (Implicitly handled by the GRU's final hidden state or we take last step)
        last_step = x[:, -1, :]
        
        rally_logits = self.rally_head(last_step)
        vol_est = self.vol_head(last_step)
        
        return rally_logits, vol_est

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentinelV2().to(device)
    dummy_input = torch.randn(2, 100, 9).to(device) # 9 features now
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        rally, vol = model(dummy_input)
    print(f"V2 Model Initialized. Output shape: {rally.shape}")
