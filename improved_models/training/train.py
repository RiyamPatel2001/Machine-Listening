"""
Training script for the improved audio-text retrieval model.

Usage:
    python training/train.py --loss infonce --embed_dim 256 --epochs 100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from training.dataset   import AudioTextDataset, MultiCaptionDataset
from encoders.projector import AudioTextModel
from training.losses    import get_loss
from training.evaluate  import retrieval_metrics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_embeddings')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--loss',          default='infonce', choices=['infonce', 'infonce_single', 'cosine', 'vicreg'])
    p.add_argument('--embed_dim',     type=int,   default=256)
    p.add_argument('--batch_size',    type=int,   default=256)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--patience',      type=int,   default=15)
    p.add_argument('--eval_every',    type=int,   default=5,   help='Compute retrieval metrics every N epochs')
    p.add_argument('--dropout_audio', type=float, default=0.3)
    p.add_argument('--dropout_text',  type=float, default=0.2)
    p.add_argument('--noise_std',     type=float, default=0.05, help='Gaussian noise on audio embeddings (0 = off)')
    p.add_argument('--save_dir',      default=None)
    return p.parse_args()


def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, device, noise_std=0.0):
    model.train()
    total_loss = 0.0
    for audio, text, clip_ids in loader:
        audio, text = audio.to(device), text.to(device)

        # Embedding augmentation: add Gaussian noise to raw audio embeddings
        if noise_std > 0.0:
            audio = audio + torch.randn_like(audio) * noise_std

        audio_emb, text_emb, logit_scale = model(audio, text)
        loss = criterion(audio_emb, text_emb, logit_scale, clip_ids)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(audio)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for audio, text, clip_ids in loader:
        audio, text = audio.to(device), text.to(device)
        audio_emb, text_emb, logit_scale = model(audio, text)
        total += criterion(audio_emb, text_emb, logit_scale, clip_ids).item() * len(audio)
    return total / len(loader.dataset)


def plot_losses(train_losses, val_losses, map_epochs, map_scores, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses,   label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2.plot(map_epochs, map_scores, marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@16 (%)')
    ax2.set_title('Validation mAP@16')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_ds = MultiCaptionDataset(f'{DATA_DIR}/audio_development.npy', f'{DATA_DIR}/text_development.npy')
    val_ds   = AudioTextDataset(f'{DATA_DIR}/audio_validation.npy',     f'{DATA_DIR}/text_validation.npy')

    print(f'Train items: {len(train_ds)} ({len(train_ds)//5} clips × 5 captions)')
    print(f'Val   items: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = AudioTextModel(
        embed_dim=args.embed_dim,
        audio_dropout=args.dropout_audio,
        text_dropout=args.dropout_text,
    ).to(device)

    criterion = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots'
    )
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f'best_{args.loss}_d{args.embed_dim}.pt')

    train_losses, val_losses = [], []
    map_epochs, map_scores   = [], []
    best_map, no_improve     = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, args.noise_std)
        vl = val_loss(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr)
        val_losses.append(vl)

        # Evaluate retrieval metrics every eval_every epochs
        if epoch % args.eval_every == 0 or epoch == 1:
            metrics = retrieval_metrics(model, val_ds, device)
            current_map = metrics['mAP@16']
            map_epochs.append(epoch)
            map_scores.append(current_map)

            print(f'Epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}  '
                  f'mAP@16={current_map:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}')

            # Early stopping on mAP@16 (what we actually care about)
            if current_map > best_map:
                best_map = current_map
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += args.eval_every

            if no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch} (best mAP@16={best_map:.2f}%)')
                break
        else:
            print(f'Epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

    # Plot
    plot_path = os.path.join(save_dir, f'losses_{args.loss}_d{args.embed_dim}.png')
    plot_losses(train_losses, val_losses, map_epochs, map_scores, plot_path)
    print(f'Plot saved to {plot_path}')

    # Final metrics from best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    metrics = retrieval_metrics(model, val_ds, device)
    print('\n--- Validation retrieval metrics (best checkpoint) ---')
    for k, v in metrics.items():
        print(f'  {k}: {v:.2f}%')

    return metrics


if __name__ == '__main__':
    main()
