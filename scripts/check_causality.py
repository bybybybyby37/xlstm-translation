import argparse
import os

import torch
from dataclasses import replace
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

from xlstm import xLSTMBlockStackConfig
from src.iwslt17_data import create_iwslt17_dataloaders, set_seed
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    p = argparse.ArgumentParser("Check decoder causality for xLSTM seq2seq")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant id used for checkpoint naming (e.g., 10, 01, 11)",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If not set, uses checkpoints/xlstm_iwslt17_en_zh_{variant}.pt",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to draw the encoder input example from (default: val)",
    )
    p.add_argument(
        "--prefix_len",
        type=int,
        default=6,
        help="Prefix length (number of target positions) to test invariance for",
    )
    p.add_argument(
        "--total_len",
        type=int,
        default=12,
        help="Total target length used in the test sequence (including BOS)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for max absolute difference on prefix logits; below this -> PASS",
    )
    return p.parse_args()


def check_decoder_causality(model, enc_out, src_mask, bos_id, eos_id, pad_id, vocab_size, device, prefix_len, total_len):
    """
    Returns (is_causal: bool, max_diff: float).

    Test: create two targets A/B with identical prefix (positions < prefix_len)
    and different suffix (positions >= prefix_len). If decoder is causal, logits
    at positions < prefix_len should be identical (up to numerical tolerance).
    """
    model.eval()
    with torch.no_grad():
        T = total_len
        t = prefix_len

        if T <= t:
            raise ValueError(f"total_len ({T}) must be > prefix_len ({t})")

        # Build two sequences [1, T]
        a = torch.full((1, T), pad_id, dtype=torch.long, device=device)
        b = torch.full((1, T), pad_id, dtype=torch.long, device=device)

        # BOS at pos 0
        a[0, 0] = bos_id
        b[0, 0] = bos_id

        # same prefix for positions 1..t-1 (so total prefix positions are 0..t-1)
        # avoid special ids (assume 0..3 reserved; if yours differs, adjust)
        low = 4
        high = max(low + 1, vocab_size - 1)

        prefix = torch.randint(low=low, high=high, size=(t - 1,), device=device)
        a[0, 1:t] = prefix
        b[0, 1:t] = prefix

        # different suffix
        a[0, t:] = torch.randint(low=low, high=high, size=(T - t,), device=device)
        b[0, t:] = torch.randint(low=low, high=high, size=(T - t,), device=device)

        logits_a = model.decode_step(enc_out, src_mask, a)  # [1, T, V]
        logits_b = model.decode_step(enc_out, src_mask, b)

        # Compare prefix positions [0:t)
        diff = (logits_a[:, :t, :] - logits_b[:, :t, :]).abs().max().item()

        return (diff < 1e-5), diff


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # ---- training cfg ----
    training_cfg = cfg.training
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    seed = int(training_cfg.seed)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- data ----
    sp, train_loader, val_loader, test_loader = create_iwslt17_dataloaders(
        vocab_size=int(dataset_cfg.vocab_size),
        max_src_len=int(dataset_cfg.max_src_len),
        max_tgt_len=int(dataset_cfg.max_tgt_len),
        batch_size=int(training_cfg.batch_size),
        num_workers=int(training_cfg.num_workers),
    )

    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()

    # Choose loader
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    # ---- model config ----
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    base_stack_cfg = from_dict(
        data_class=xLSTMBlockStackConfig,
        data=model_cfg_dict,
        config=DaciteConfig(strict=True),
    )

    max_src_len = int(dataset_cfg.max_src_len)
    max_tgt_len = int(dataset_cfg.max_tgt_len)

    enc_cfg = replace(base_stack_cfg, context_length=max_src_len)
    dec_cfg = replace(base_stack_cfg, context_length=max_tgt_len)

    model = XlstmSeq2Seq(enc_cfg=enc_cfg, dec_cfg=dec_cfg, vocab_size=vocab_size).to(device)

    # ---- load checkpoint ----
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join("checkpoints", f"xlstm_iwslt17_en_zh_{args.variant}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # ---- get one example src ----
    src, tgt_in, tgt_out = next(iter(loader))
    src_i = src[0:1].to(device)  # [1, L]

    # ---- encode ----
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_i)

    # ---- causality check ----
    ok, max_diff = check_decoder_causality(
        model=model,
        enc_out=enc_out,
        src_mask=src_mask,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        vocab_size=vocab_size,
        device=device,
        prefix_len=args.prefix_len,
        total_len=args.total_len,
    )

    print("============================================================")
    print(f"Causality check on decoder (prefix_len={args.prefix_len}, total_len={args.total_len})")
    print(f"Max |logits_a - logits_b| on prefix positions: {max_diff:.6e}")
    print(f"Tolerance: {args.tol:.2e}")
    if max_diff <= args.tol:
        print("RESULT: PASS (decoder appears causal under this test)")
    else:
        print("RESULT: FAIL (decoder likely uses future tokens; consider step-wise decoding / causal enforcement)")
    print("============================================================")


if __name__ == "__main__":
    main()
