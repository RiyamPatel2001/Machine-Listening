import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioProjector(nn.Module):
    """
    Maps CNN14 embeddings (2048-d, unnormalized) → shared embedding space (embed_dim-d, L2-normalized).
    Three-layer MLP with BN + ReLU + Dropout; final L2 norm.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, embed_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TextProjector(nn.Module):
    """
    Maps SBERT embeddings (768-d, already L2-normalized) → shared embedding space (embed_dim-d, L2-normalized).
    Two-layer MLP; keeps capacity moderate since SBERT embeddings are already high quality.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, embed_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class AudioTextModel(nn.Module):
    """Combined bi-encoder model holding both projectors."""

    def __init__(self, audio_input_dim: int = 2048, text_input_dim: int = 768, embed_dim: int = 512,
                 audio_hidden: int = 1024, text_hidden: int = 512,
                 audio_dropout: float = 0.3, text_dropout: float = 0.2):
        super().__init__()
        self.audio_proj = AudioProjector(audio_input_dim, audio_hidden, embed_dim, audio_dropout)
        self.text_proj  = TextProjector(text_input_dim, text_hidden, embed_dim, text_dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # log(1/0.07)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(audio)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text)

    def forward(self, audio: torch.Tensor, text: torch.Tensor):
        audio_emb = self.encode_audio(audio)
        text_emb  = self.encode_text(text)
        return audio_emb, text_emb, self.logit_scale.exp()
