import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPositiveInfoNCE(nn.Module):
    """
    Symmetric InfoNCE with multi-positive support.

    When multiple items in a batch share the same clip_id (i.e. different
    captions of the same audio clip), all are treated as positives rather
    than negatives. Degenerates to standard symmetric InfoNCE when each
    clip appears exactly once (clip_id == row index).

    Loss per anchor i (audio→text direction):
        L_i = log( sum_k exp(s_ik) ) - (1/|P_i|) * sum_{p in P_i} s_ip
    where P_i = {j : clip_id[j] == clip_id[i]}.
    """

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor,
                logit_scale: torch.Tensor, clip_ids=None) -> torch.Tensor:
        logit_scale = logit_scale.clamp(max=100.0)
        logits = logit_scale * audio_emb @ text_emb.T   # (B, B)
        B      = audio_emb.shape[0]
        device = audio_emb.device

        if clip_ids is None:
            pos_mask = torch.eye(B, device=device)
        else:
            clip_ids = clip_ids.to(device)
            pos_mask = (clip_ids.unsqueeze(1) == clip_ids.unsqueeze(0)).float()  # (B, B)

        # Audio → Text
        log_Z_a    = torch.logsumexp(logits, dim=1)                          # (B,)
        avg_pos_a  = (logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)   # (B,)
        loss_a     = (log_Z_a - avg_pos_a).mean()

        # Text → Audio
        logits_t   = logits.T
        log_Z_t    = torch.logsumexp(logits_t, dim=1)
        avg_pos_t  = (logits_t * pos_mask.T).sum(dim=1) / pos_mask.T.sum(dim=1)
        loss_t     = (log_Z_t - avg_pos_t).mean()

        return (loss_a + loss_t) / 2


class SymmetricInfoNCE(nn.Module):
    """Legacy single-positive InfoNCE, kept for comparison runs."""

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor,
                logit_scale: torch.Tensor, clip_ids=None) -> torch.Tensor:
        logit_scale = logit_scale.clamp(max=100.0)
        logits_per_audio = logit_scale * audio_emb @ text_emb.T
        logits_per_text  = logits_per_audio.T
        labels = torch.arange(len(audio_emb), device=audio_emb.device)
        loss_a = F.cross_entropy(logits_per_audio, labels)
        loss_t = F.cross_entropy(logits_per_text,  labels)
        return (loss_a + loss_t) / 2


class CosineLoss(nn.Module):
    """
    Mean cosine loss over all positive pairs: 1 - cos_sim(audio_i, text_i).
    Simple and stable but ignores negatives — best combined with InfoNCE.
    """

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale=None) -> torch.Tensor:
        cos_sim = (audio_emb * text_emb).sum(dim=-1)   # (B,)  embeddings are already L2-normalized
        return (1 - cos_sim).mean()


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization (Bardes et al. 2022).
    Encourages:
      - Invariance: audio and text embeddings of the same pair should be close
      - Variance: each dimension should have unit variance across the batch
      - Covariance: off-diagonal terms of the covariance matrix should be zero
    """

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale=None) -> torch.Tensor:
        B, D = audio_emb.shape

        # Invariance: MSE between paired embeddings
        inv_loss = F.mse_loss(audio_emb, text_emb)

        # Variance: push std of each dim toward 1
        def var_loss(z):
            std = z.std(dim=0)
            return F.relu(1 - std).mean()

        var_l = var_loss(audio_emb) + var_loss(text_emb)

        # Covariance: penalize off-diagonal covariance
        def cov_loss(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)
            off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            return off_diag / D

        cov_l = cov_loss(audio_emb) + cov_loss(text_emb)

        return self.sim_weight * inv_loss + self.var_weight * var_l + self.cov_weight * cov_l


def get_loss(name: str) -> nn.Module:
    losses = {
        'infonce':        MultiPositiveInfoNCE,
        'infonce_single': SymmetricInfoNCE,
        'cosine':         CosineLoss,
        'vicreg':         VICRegLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(losses)}")
    return losses[name]()
