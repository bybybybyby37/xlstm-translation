import os
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm

from src.iwslt17_data import (
    set_seed,
    collate_translation_batch,
    IWSLT17EnZhDataset,
)


class SyntheticEnZhTxtDataset(Dataset):
    """
    Synthetic file format:
      line 0: English
      line 1: Chinese
      line 2: English
      line 3: Chinese
      ...
    """

    def __init__(
        self,
        txt_path: str,
        sp: spm.SentencePieceProcessor,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Synthetic txt not found: {txt_path}")

        self.sp = sp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.pad_id = sp.pad_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()

        self.pairs: List[Tuple[str, str]] = self._load_pairs(txt_path)

    @staticmethod
    def _load_pairs(txt_path: str) -> List[Tuple[str, str]]:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]

        if len(lines) % 2 != 0:
            raise ValueError(f"Synthetic file must have even number of lines, got {len(lines)}")

        pairs = []
        for i in range(0, len(lines), 2):
            en = lines[i].strip()
            zh = lines[i + 1].strip()
            if en and zh:
                pairs.append((en, zh))

        if not pairs:
            raise ValueError(f"No valid pairs loaded from {txt_path}")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id] + self.sp.encode(text, out_type=int)[: max_len - 2] + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        en, zh = self.pairs[idx]
        src_ids = self._encode(en, self.max_src_len)
        tgt_ids = self._encode(zh, self.max_tgt_len)
        return src_ids, tgt_ids


class MixedAnchorSynthDataset(Dataset):
    """
    Anchor-on-synthetic mixing:
      - Use ALL synthetic examples exactly once per epoch (no oversampling).
      - Use ratio_iwslt * n_synth IWSLT examples per epoch.
      - IWSLT examples are taken by cycling through a fixed permutation to ensure coverage across epochs.

    Effective epoch size:
      n_epoch = n_synth + ratio_iwslt * n_synth = (ratio_iwslt + 1) * n_synth
    """

    def __init__(
        self,
        iwslt_train: Dataset,
        synth: Dataset,
        ratio_iwslt: int = 10,
        seed: int = 1337,
        epoch: int = 1,
    ):
        self.iwslt_train = iwslt_train
        self.synth = synth
        self.ratio_iwslt = ratio_iwslt
        self.seed = seed
        self.epoch = epoch

        self.n_iwslt = len(iwslt_train)
        self.n_synth = len(synth)

        # IWSLT per epoch is limited by synthetic (no need to exceed full iwslt)
        self.n_iwslt_take = min(self.n_iwslt, self.n_synth * self.ratio_iwslt)

        # Build a deterministic global permutation of IWSLT indices once
        rng_global = np.random.RandomState(self.seed)
        self.iwslt_perm = rng_global.permutation(self.n_iwslt).tolist()

        # Precompute mapping for this epoch
        self.mapping: List[Tuple[int, int]] = self._build_mapping_for_epoch(epoch)

    def _build_mapping_for_epoch(self, epoch: int) -> List[Tuple[int, int]]:
        # Epoch-specific RNG (for shuffling order + synth permutation)
        rng = np.random.RandomState(self.seed + epoch)

        # 1) synthetic: use all once, but shuffled each epoch
        synth_idx = rng.permutation(self.n_synth).tolist()

        # 2) iwslt: take a slice from the global permutation, cycling by epoch
        #    start advances deterministically each epoch
        stride = self.n_iwslt_take
        start = ((epoch - 1) * stride) % self.n_iwslt

        iwslt_idx = []
        if start + self.n_iwslt_take <= self.n_iwslt:
            iwslt_idx = self.iwslt_perm[start : start + self.n_iwslt_take]
        else:
            part1 = self.iwslt_perm[start:]
            part2 = self.iwslt_perm[: (start + self.n_iwslt_take) - self.n_iwslt]
            iwslt_idx = part1 + part2

        # 3) combine then shuffle to mix within epoch
        mapping: List[Tuple[int, int]] = [(0, i) for i in iwslt_idx] + [(1, j) for j in synth_idx]
        rng.shuffle(mapping)
        return mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        src_flag, j = self.mapping[idx]
        if src_flag == 0:
            return self.iwslt_train[j]
        return self.synth[j]


def load_spm(spm_model_path: str) -> spm.SentencePieceProcessor:
    if not os.path.exists(spm_model_path):
        raise FileNotFoundError(f"SPM model not found: {spm_model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    return sp


def prepare_iwslt17_and_synth(
    *,
    spm_model_path: str,
    synthetic_txt_path: str,
    max_src_len: int,
    max_tgt_len: int,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    """
    Prepare:
      - sp
      - iwslt_train dataset (NOT loader)
      - synth dataset
      - val_loader, test_loader
      - collate_fn
    Train loader will be created per-epoch in training script (so no epoch-state propagation issues).
    """
    set_seed(seed)

    print("[Data] loading IWSLT2017 en-zh dataset...")
    ds = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)

    sp = load_spm(spm_model_path)
    pad_id = sp.pad_id()
    collate_fn: Callable = lambda batch: collate_translation_batch(batch, pad_id=pad_id)

    iwslt_train = IWSLT17EnZhDataset(ds["train"], sp, max_src_len, max_tgt_len, "en", "zh")
    val_set = IWSLT17EnZhDataset(ds["validation"], sp, max_src_len, max_tgt_len, "en", "zh")
    test_set = IWSLT17EnZhDataset(ds["test"], sp, max_src_len, max_tgt_len, "en", "zh")

    synth_set = SyntheticEnZhTxtDataset(synthetic_txt_path, sp, max_src_len, max_tgt_len)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(
        f"[Data] AnchorSynth ready: iwslt_train={len(iwslt_train)}, synth_total={len(synth_set)}"
    )
    return sp, iwslt_train, synth_set, val_loader, test_loader, collate_fn
