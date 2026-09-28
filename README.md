# Language-Based Audio Retrieval

Audio-text retrieval on the Clotho v2 dataset: given a text caption, retrieve and rank the audio
clips it best describes.

## Task

Given a text caption, retrieve and rank the audio clips it best describes, using the
[Clotho v2](https://zenodo.org/record/4783391) dataset (6,974 clips, 5 captions each; 3,839 dev /
1,045 validation / 1,045 test clips). Evaluated with R@1, R@5, R@10, and mAP.

## Structure

- **`models/`** — Original architecture experiments (write-up: [`report/`](report/)). A bi-encoder
  setup (pretrained CNN14 audio encoder + frozen Sentence-BERT text encoder, InfoNCE loss) was used
  as the baseline, and several custom audio encoders were tried in its place, trained end-to-end on
  mel-spectrograms:
  - `CRNN.py` — 3D CNN + bidirectional GRU
  - `TCN.py` — dilated temporal convolutional network
  - `spec2vec_encoder.py` — simple MLP over the flattened spectrogram
  - `MelSpectogram_CNN_BEATS.py` / `Multiscale_MelSpectogram_CNN_BEATS.py` — small CNN combined
    with a pretrained wav2vec2 model, on normal and multiscale mel-spectrograms respectively
  - `plots/` — training/validation loss curves for each encoder

  **Reported results (mAP@10):** best custom encoder (CRNN, normal mel, InfoNCE) = 0.108;
  pretrained wav2vec2/BEATs-style encoder = 0.152; CNN14+SBERT baseline = 0.222. Multiscale
  spectrograms did not outperform normal ones for any encoder.

- **`improved_models/`** — A follow-up redesign of the retrieval pipeline, moving from end-to-end
  raw-audio encoders to lightweight learned projectors over precomputed embeddings:
  - `data_embeddings/` — precomputed CNN14 audio embeddings (2048-d) and Sentence-BERT text
    embeddings (768-d × 5 captions/clip) for the dev/validation/eval splits
  - `encoders/projector.py` — `AudioProjector` / `TextProjector` MLPs mapping both modalities into
    a shared, L2-normalized embedding space, plus a CLIP-style learnable logit scale
  - `training/losses.py` — a small loss library: multi-positive InfoNCE (treats all 5 captions of
    a clip as positives, rather than as negatives of each other), single-positive InfoNCE, cosine
    loss, and VICReg
  - `training/dataset.py`, `train.py`, `evaluate.py` — multi-caption training loop with cosine LR
    warmup, embedding-noise augmentation, and early stopping on validation mAP@16 (rather than
    validation loss); retrieval metrics (R@1/R@5/R@10/mAP@16)
  - `plots/` — trained checkpoints and loss/mAP curves for `embed_dim=256` and `embed_dim=512`

  **Result:** ~23.5–24% mAP@16 at both embedding dims, matching/exceeding the original baseline's
  reported 22.2% mAP@10 and well above the custom encoders explored in `models/`.

- **`report/`** — Final project report (PDF).

## Usage

```bash
pip install -r requirements.txt
cd improved_models
python training/train.py --loss infonce --embed_dim 256 --epochs 100
```

`models/*.py` contains standalone `nn.Module` encoder definitions from the original experiments
(not wired into a runnable training script in this repo).
