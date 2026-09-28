import torch
import numpy as np


@torch.no_grad()
def retrieval_metrics(model, dataset, device: str, batch_size: int = 256) -> dict:
    """
    Text→Audio retrieval evaluation.

    For each of the 5*N text queries, retrieve from N audio clips and
    compute R@1, R@5, R@10, mAP@10.

    Ground truth: caption i*5+j belongs to audio clip i.
    """
    model.eval()

    # Encode all audio clips
    all_audio_raw = dataset.get_all_audio().to(device)  # (N, 2048)
    audio_embs = []
    for i in range(0, len(all_audio_raw), batch_size):
        audio_embs.append(model.encode_audio(all_audio_raw[i:i+batch_size]))
    audio_embs = torch.cat(audio_embs, dim=0)           # (N, D), L2-normalized

    # Encode all text queries (5 per clip)
    all_text_raw = dataset.get_all_text().to(device)    # (N, 5, 768)
    N = all_text_raw.shape[0]
    text_flat = all_text_raw.view(N * 5, -1)            # (5N, 768)
    text_embs = []
    for i in range(0, len(text_flat), batch_size):
        text_embs.append(model.encode_text(text_flat[i:i+batch_size]))
    text_embs = torch.cat(text_embs, dim=0)             # (5N, D), L2-normalized

    # Similarity matrix: (5N, N)
    sim = text_embs @ audio_embs.T                      # cosine sim (already normalized)
    sim_np = sim.cpu().numpy()

    # Ground truth: query i*5+j → audio i
    labels = np.repeat(np.arange(N), 5)                 # (5N,)

    r1, r5, r10, maps = [], [], [], []
    for q_idx in range(N * 5):
        gt = labels[q_idx]
        ranked = np.argsort(sim_np[q_idx])[::-1]        # descending

        r1.append(int(gt in ranked[:1]))
        r5.append(int(gt in ranked[:5]))
        r10.append(int(gt in ranked[:10]))

        # AP@16 — competition metric
        ap = 0.0
        hits = 0
        for k, r in enumerate(ranked[:16], 1):
            if r == gt:
                hits += 1
                ap += hits / k
        maps.append(ap)

    return {
        'R@1':    np.mean(r1)   * 100,
        'R@5':    np.mean(r5)   * 100,
        'R@10':   np.mean(r10)  * 100,
        'mAP@16': np.mean(maps) * 100,
    }
