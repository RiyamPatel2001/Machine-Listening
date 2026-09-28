import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.block(x) + self.residual(x)

class TCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(1, 64, dilation=1),
            TCNBlock(64, 128, dilation=2),
            TCNBlock(128, 256, dilation=4),
            TCNBlock(256, 512, dilation=8),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(512, 512)

    def forward(self, x):  # x: (B, 1, H, W)
        # Reshape to (B, num_scales, H * W)
        x = x.view(x.size(0), x.size(2), -1)
        # Average across scales
        x = x.mean(1).unsqueeze(1) # (B, 1, H * W)
        x = self.tcn(x).squeeze(-1)             # (B, 512)
        return self.proj(x)
