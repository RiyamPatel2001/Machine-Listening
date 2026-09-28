import numpy as np
import torch
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """
    Single-caption dataset — used for validation loss monitoring and evaluation.
    Returns one randomly sampled caption per audio clip per call.
    clip_id == idx (each clip is unique), so multi-positive loss degenerates
    to standard InfoNCE here.
    """

    def __init__(self, audio_path: str, text_path: str):
        audio_dict = np.load(audio_path, allow_pickle=True).item()
        text_dict  = np.load(text_path,  allow_pickle=True).item()

        self.keys  = sorted(audio_dict.keys())
        self.audio = [torch.tensor(audio_dict[k], dtype=torch.float32) for k in self.keys]
        self.text  = [torch.tensor(text_dict[k],  dtype=torch.float32) for k in self.keys]
        # self.audio[i]: (2048,)
        # self.text[i]:  (5, 768)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        audio   = self.audio[idx]
        cap_idx = torch.randint(0, 5, (1,)).item()
        text    = self.text[idx][cap_idx]
        return audio, text, idx          # idx serves as clip_id

    def get_all_audio(self) -> torch.Tensor:
        return torch.stack(self.audio)   # (N, 2048)

    def get_all_text(self) -> torch.Tensor:
        return torch.stack(self.text)    # (N, 5, 768)


class MultiCaptionDataset(Dataset):
    """
    Multi-caption training dataset — expands each audio clip into 5 separate
    training items (one per caption), giving 19,195 items from 3,839 clips.

    Returns clip_id (integer index of the source audio clip) so the loss
    function can identify which items in a batch are positive pairs for
    the same audio.
    """

    def __init__(self, audio_path: str, text_path: str):
        audio_dict = np.load(audio_path, allow_pickle=True).item()
        text_dict  = np.load(text_path,  allow_pickle=True).item()

        keys = sorted(audio_dict.keys())
        self.items = []   # list of (audio_emb, text_emb, clip_id)

        for clip_id, key in enumerate(keys):
            audio = torch.tensor(audio_dict[key], dtype=torch.float32)   # (2048,)
            texts = torch.tensor(text_dict[key],  dtype=torch.float32)   # (5, 768)
            for cap_idx in range(5):
                self.items.append((audio, texts[cap_idx], clip_id))

        # Keep grouped copies for evaluation helpers
        self._audio = [torch.tensor(audio_dict[k], dtype=torch.float32) for k in keys]
        self._text  = [torch.tensor(text_dict[k],  dtype=torch.float32) for k in keys]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio, text, clip_id = self.items[idx]
        return audio, text, clip_id

    def get_all_audio(self) -> torch.Tensor:
        return torch.stack(self._audio)  # (N, 2048)

    def get_all_text(self) -> torch.Tensor:
        return torch.stack(self._text)   # (N, 5, 768)
