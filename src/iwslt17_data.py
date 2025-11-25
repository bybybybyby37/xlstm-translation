import os
from typing import List, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
import numpy as np


def set_seed(seed: int = 1337):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- SentencePiece ----------

def _train_sentencepiece_model(
    texts: List[str],
    model_prefix: str,
    vocab_size: int = 8000,
):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    input_txt = model_prefix + ".txt"

    with open(input_txt, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.Train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )


def load_or_train_spm_for_iwslt17(
    spm_dir: str = "spm",
    vocab_size: int = 8000,
) -> spm.SentencePieceProcessor:
    """
    read or train IWSLT17 En-Zh 's shared sentencepiece model.
    """
    os.makedirs(spm_dir, exist_ok=True)
    model_prefix = os.path.join(spm_dir, f"iwslt17_en_zh_{vocab_size}")
    model_file = model_prefix + ".model"

    if not os.path.exists(model_file):
        print("[SPM] training new SentencePiece model...")
        ds = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh")
        texts: List[str] = []
        for ex in ds["train"]:
            tr = ex["translation"]
            texts.append(tr["en"])
            texts.append(tr["zh"])
        _train_sentencepiece_model(texts, model_prefix, vocab_size=vocab_size)
    else:
        print(f"[SPM] using existing SentencePiece model: {model_file}")

    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


# ---------- Dataset & collate ----------

class IWSLT17EnZhDataset(Dataset):
    def __init__(
        self,
        split_data,
        sp: spm.SentencePieceProcessor,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        src_lang: str = "en",
        tgt_lang: str = "zh",
    ):
        self.data = split_data
        self.sp = sp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.pad_id = sp.pad_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()

    def __len__(self):
        return len(self.data)

    def _encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id] + self.sp.encode(text, out_type=int)[: max_len - 2] + [
            self.eos_id
        ]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.data[idx]["translation"]
        src = ex[self.src_lang]
        tgt = ex[self.tgt_lang]

        src_ids = self._encode(src, self.max_src_len)
        tgt_ids = self._encode(tgt, self.max_tgt_len)
        return src_ids, tgt_ids


def collate_translation_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
):
    """
    pad the variable lens to batch: 
    return:
      src_batch: [B, S]
      tgt_in:    [B, T]  (decoder input)
      tgt_out:   [B, T]  (prediction)
    """
    src_lens = [len(src) for src, _ in batch]
    tgt_lens = [len(tgt) for _, tgt in batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    B = len(batch)
    src_batch = torch.full((B, max_src), pad_id, dtype=torch.long)
    tgt_in = torch.full((B, max_tgt - 1), pad_id, dtype=torch.long)
    tgt_out = torch.full((B, max_tgt - 1), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_len = len(src)
        tgt_len = len(tgt)

        src_batch[i, :src_len] = src
        tgt_in[i, : tgt_len - 1] = tgt[:-1]   # remove last token
        tgt_out[i, : tgt_len - 1] = tgt[1:]   # remove token

    return src_batch, tgt_in, tgt_out


def create_iwslt17_dataloaders(
    vocab_size: int = 8000,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,  # Windows → 0 most safe
):
    """
    return (sp, train_loader, val_loader, test_loader)
    """
    set_seed(1337)

    print("[Data] loading IWSLT2017 en-zh dataset...")
    dataset_dict = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)

    sp = load_or_train_spm_for_iwslt17(vocab_size=vocab_size)

    train_set = IWSLT17EnZhDataset(
        dataset_dict["train"], sp, max_src_len, max_tgt_len, "en", "zh"
    )
    val_set = IWSLT17EnZhDataset(
        dataset_dict["validation"], sp, max_src_len, max_tgt_len, "en", "zh"
    )
    test_set = IWSLT17EnZhDataset(
        dataset_dict["test"], sp, max_src_len, max_tgt_len, "en", "zh"
    )

    pad_id = sp.pad_id()
    collate_fn: Callable = lambda batch: collate_translation_batch(batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return sp, train_loader, val_loader, test_loader
