import os
from typing import List, Tuple, Callable, Optional

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
    Read synthetic parallel pairs from a txt file:
      line 0: English
      line 1: Chinese
      line 2: English
      line 3: Chinese
      ...
    Returns (src_ids, tgt_ids) with BOS/EOS, truncated to max lens.
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

        self.txt_path = txt_path
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
            raise ValueError(
                f"Synthetic file must have even number of lines (en/zh pairs). "
                f"Got {len(lines)} lines: {txt_path}"
            )

        pairs = []
        for i in range(0, len(lines), 2):
            en = lines[i].strip()
            zh = lines[i + 1].strip()
            if en and zh:
                pairs.append((en, zh))

        if len(pairs) == 0:
            raise ValueError(f"No valid (en, zh) pairs loaded from: {txt_path}")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id] + self.sp.encode(text, out_type=int)[: max_len - 2] + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        en, zh = self.pairs[idx]
        src_ids = self._encode(en, self.max_src_len)
        tgt_ids = self._encode(zh, self.max_tgt_len)
        return src_ids, tgt_ids


class MixedTranslationDataset(Dataset):
    """
    Deterministic mixed dataset with approximate ratio:
        iwslt : synth = ratio_iwslt : ratio_synth

    Implementation choice (reproducible, minimal intrusion):
      - include every IWSLT sample once per epoch-length
      - add N_synth samples sampled with replacement to match ratio
      - DataLoader shuffle with seeded generator handles ordering

    If n_iwslt = 200k and ratio=10:1 => n_synth_needed = ceil(200k/10)=20k
    """

    def __init__(
        self,
        iwslt_train: Dataset,
        synth: Dataset,
        ratio_iwslt: int = 10,
        ratio_synth: int = 1,
        seed: int = 1337,
    ):
        if ratio_iwslt <= 0 or ratio_synth <= 0:
            raise ValueError("ratio_iwslt and ratio_synth must be positive integers.")

        self.iwslt_train = iwslt_train
        self.synth = synth
        self.ratio_iwslt = ratio_iwslt
        self.ratio_synth = ratio_synth
        self.seed = seed

        self.n_iwslt = len(iwslt_train)
        self.n_synth = len(synth)

        # how many synth examples to add to approximate ratio
        self.n_synth_needed = int(np.ceil(self.n_iwslt * (ratio_synth / ratio_iwslt)))
        self.mapping: List[Tuple[int, int]] = self._build_mapping()

    def _build_mapping(self) -> List[Tuple[int, int]]:
        # (source_flag, idx) where source_flag: 0=iwslt, 1=synth
        mapping: List[Tuple[int, int]] = [(0, i) for i in range(self.n_iwslt)]

        rng = np.random.RandomState(self.seed)
        synth_indices = rng.randint(low=0, high=self.n_synth, size=self.n_synth_needed).tolist()
        mapping.extend((1, j) for j in synth_indices)

        return mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        src_flag, j = self.mapping[idx]
        if src_flag == 0:
            return self.iwslt_train[j]  # (src_ids, tgt_ids)
        return self.synth[j]  # (src_ids, tgt_ids)


def load_spm_from_file(spm_model_path: str) -> spm.SentencePieceProcessor:
    if not os.path.exists(spm_model_path):
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    return sp


def _worker_init_fn(worker_id: int, base_seed: int):
    # Ensure each worker has deterministic, distinct seed
    set_seed(base_seed + worker_id)


def create_mixed_iwslt17_dataloaders(
    *,
    spm_model_path: str,
    synthetic_txt_path: str,
    vocab_size: int = 8000,          # kept for interface consistency; spm file is authoritative
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 1337,
    ratio_iwslt: int = 10,
    ratio_synth: int = 1,
):
    """
    Return (sp, train_loader_mixed, val_loader, test_loader)
    - Train loader is mixed (IWSLT train + synthetic)
    - Val/Test are pure IWSLT (same as original)
    """
    set_seed(seed)

    print("[Data] loading IWSLT2017 en-zh dataset...")
    dataset_dict = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)

    # IMPORTANT: use your existing SPM to ensure same vocab & ids as original training
    sp = load_spm_from_file(spm_model_path)

    # IWSLT datasets
    iwslt_train = IWSLT17EnZhDataset(
        dataset_dict["train"], sp, max_src_len, max_tgt_len, "en", "zh"
    )
    val_set = IWSLT17EnZhDataset(
        dataset_dict["validation"], sp, max_src_len, max_tgt_len, "en", "zh"
    )
    test_set = IWSLT17EnZhDataset(
        dataset_dict["test"], sp, max_src_len, max_tgt_len, "en", "zh"
    )

    # Synthetic dataset
    synth_set = SyntheticEnZhTxtDataset(
        synthetic_txt_path, sp, max_src_len=max_src_len, max_tgt_len=max_tgt_len
    )

    mixed_train = MixedTranslationDataset(
        iwslt_train=iwslt_train,
        synth=synth_set,
        ratio_iwslt=ratio_iwslt,
        ratio_synth=ratio_synth,
        seed=seed,
    )

    pad_id = sp.pad_id()
    collate_fn: Callable = lambda batch: collate_translation_batch(batch, pad_id=pad_id)

    # deterministic shuffle generator
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        mixed_train,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=(lambda wid: _worker_init_fn(wid, seed)) if num_workers > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=(lambda wid: _worker_init_fn(wid, seed)) if num_workers > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=(lambda wid: _worker_init_fn(wid, seed)) if num_workers > 0 else None,
        drop_last=False,
    )

    print(
        f"[Data] Mixed train size = {len(mixed_train)} "
        f"(iwslt={len(iwslt_train)}, synth_added={mixed_train.n_synth_needed}, synth_total={len(synth_set)}), "
        f"ratio={ratio_iwslt}:{ratio_synth}"
    )

    return sp, train_loader, val_loader, test_loader
