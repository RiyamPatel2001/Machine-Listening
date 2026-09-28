import torch
import torch.nn as nn


class CRNNEncoder(nn.Module):
    def __init__(self, input_size=(64, 128), hidden_size=256, num_layers=3):
        super(CRNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),  # Changed the input channel to 3
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # Changed to Conv3d
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # Changed to Conv3d
            nn.ReLU(),
            nn.BatchNorm3d(256)  # Changed to BatchNorm3d
        )
        # The input size to the GRU should be calculated based on the output shape of the CNN
        # Given input_size = (64, 128), output of CNN will have spatial dimensions (64 // 4, 128 // 4) = (16, 32)
        # The input size to GRU should be (batch_size, sequence_length, input_size)
        # Here, sequence_length is 32 (time dimension), and input_size is 256 * 16 (feature dimension)

        # Correcting the input size for GRU to match the CNN output
        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size, # Changed input_size to 256
                          num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.proj = nn.Linear(2 * hidden_size, 512)
        self.proj_to_text_dim = nn.Linear(512, 768) # Project to 768 dimensions


    def forward(self, x):  # x: (B, 1, num_scales, H, W)
        # x shape is (B, 3, 1, H, W)
        # permute to (B, 3, H, W, 1)
        x = x.permute(0, 1, 3, 4, 2)
        # permute to (B, 3, 1, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.cnn(x)  # (B, 256, num_scales, H/4, W/4)
        # Average across the scales dimension
        x = x.mean(2)  # (B, 256, H/4, W/4)
        # Reshape the tensor for GRU input: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), x.size(2) * x.size(3), x.size(1)) # (B, W/4, H/4, C) -> (B, H/4 * W/4, C)
        output, _ = self.gru(x)
        x = self.proj(output[:, -1, :])  # (B, 512)
        # Apply the projection layer
        x = self.proj_to_text_dim(x) # (B, 768)
        return x
