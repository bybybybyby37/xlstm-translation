'''
python -m scripts.interactive_translate2 \
  --config config/iwslt17_xlstm11_mix10to1_anchor_synth.yaml \
  --ckpt checkpoints/xlstm_iwslt17_en_zh_11_anchorS_mix10to1_bestbleu_currentBEST.pt \
  --cli
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import replace
from typing import Optional, Tuple, Any, Dict

import torch
import sentencepiece as spm
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

from xlstm import xLSTMBlockStackConfig
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    p = argparse.ArgumentParser("Interactive EN->ZH translator (xLSTM IWSLT17)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"])
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--len_penalty", type=float, default=0.6)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--cli", action="store_true")
    return p.parse_args()


def pick_device(arg: Optional[str]) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_plain_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    return OmegaConf.to_container(x, resolve=True) if OmegaConf.is_config(x) else dict(x)


def _find_first(cfg: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            return cfg[k]
        if OmegaConf.is_config(cfg) and k in cfg:
            return cfg[k]
    return None


def load_cfg_with_base(config_path: str):
    """
    Load YAML and if it contains `base_config: <path>`, load that base yaml and merge:
    merged = merge(base, current) so current overrides base.
    """
    cfg = OmegaConf.load(config_path)

    base_ref = cfg.get("base_config")
    if isinstance(base_ref, str) and base_ref.strip():
        base_ref = base_ref.strip()

        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        cand_paths = [
            base_ref,
            os.path.join(cfg_dir, base_ref),
        ]
        base_path = None
        for p in cand_paths:
            if os.path.exists(p):
                base_path = p
                break

        if base_path is None:
            raise FileNotFoundError(
                "base_config is a string but cannot resolve it to a yaml file.\n"
                f"  base_config={base_ref}\n"
                f"  tried: {cand_paths}"
            )

        base_cfg = OmegaConf.load(base_path)
        merged = OmegaConf.merge(base_cfg, cfg)  # base first, then overrides
        # attach for debug
        merged._base_path = base_path
        return merged

    return cfg


def _resolve_spm_path(sp_model_path: str, ckpt_path: str, config_path: str) -> str:
    sp_model_path = (sp_model_path or "").strip()
    if not sp_model_path:
        raise RuntimeError("Empty sp_model_path in checkpoint.")

    if os.path.isabs(sp_model_path) and os.path.exists(sp_model_path):
        return sp_model_path
    if os.path.exists(sp_model_path):
        return sp_model_path

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cand = os.path.join(ckpt_dir, sp_model_path)
    if os.path.exists(cand):
        return cand

    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    cand = os.path.join(cfg_dir, sp_model_path)
    if os.path.exists(cand):
        return cand

    raise FileNotFoundError(
        f"SentencePiece model not found.\n"
        f"  sp_model_path(from ckpt)={sp_model_path}\n"
        f"  tried: {sp_model_path}\n"
        f"         {os.path.join(ckpt_dir, sp_model_path)}\n"
        f"         {os.path.join(cfg_dir, sp_model_path)}"
    )


def load_sentencepiece_from_ckpt(ckpt: dict, ckpt_path: str, config_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    if "sp_model" in ckpt and ckpt["sp_model"] is not None:
        sp.LoadFromSerializedProto(ckpt["sp_model"])
        return sp
    if "sp_model_path" in ckpt and ckpt["sp_model_path"] is not None:
        spm_path = _resolve_spm_path(str(ckpt["sp_model_path"]), ckpt_path, config_path)
        sp.Load(spm_path)
        return sp
    raise RuntimeError("Checkpoint has neither 'sp_model' nor 'sp_model_path'.")


def build_model_from_config(cfg, vocab_size: int, pad_id: int) -> Tuple[XlstmSeq2Seq, int, int]:
    """
    After merging base_config, we expect:
      cfg.dataset.{max_src_len, max_tgt_len}
      cfg.model  (xLSTMBlockStackConfig-compatible)
    """
    if "dataset" not in cfg:
        raise RuntimeError(f"Config missing 'dataset'. top keys={list(_to_plain_dict(cfg).keys())}")
    if "model" not in cfg:
        raise RuntimeError(
            f"Config missing 'model' even after base_config merge.\n"
            f"top keys={list(_to_plain_dict(cfg).keys())}"
        )

    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    max_src_len = int(dataset_cfg.max_src_len)
    max_tgt_len = int(dataset_cfg.max_tgt_len)

    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

    base_stack_cfg = from_dict(
        data_class=xLSTMBlockStackConfig,
        data=model_cfg_dict,
        config=DaciteConfig(strict=False),
    )

    enc_cfg = replace(base_stack_cfg, context_length=max_src_len)
    dec_cfg = replace(base_stack_cfg, context_length=max_tgt_len)

    model = XlstmSeq2Seq(
        vocab_size=vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        pad_id=pad_id,
        enc_cfg=enc_cfg,
        dec_cfg=dec_cfg,
    )
    return model, max_src_len, max_tgt_len


def clean_piece_ids(ids, bos_id, eos_id, pad_id):
    out = []
    for x in ids:
        if x == pad_id:
            continue
        if x == eos_id:
            break
        if x == bos_id:
            continue
        out.append(int(x))
    return out


@torch.no_grad()
def translate_one(
    model: XlstmSeq2Seq,
    sp: spm.SentencePieceProcessor,
    text: str,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_src_len: int,
    max_tgt_len: int,
    decode: str,
    beam_size: int,
    len_penalty: float,
    max_len: Optional[int],
) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    src_ids = [bos_id] + sp.encode(text, out_type=int) + [eos_id]
    src_ids = src_ids[:max_src_len]
    if len(src_ids) < max_src_len:
        src_ids += [pad_id] * (max_src_len - len(src_ids))

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_out, src_mask = model.encode(src)

    use_max_len = int(max_len) if (max_len is not None) else int(max_tgt_len)

    if decode == "beam":
        gen = model.beam_decode(
            enc_out,
            src_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=use_max_len,
            beam_size=beam_size,
            len_penalty=len_penalty,
        )
    else:
        gen = model.greedy_decode(
            enc_out,
            src_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=use_max_len,
        )

    hyp_ids = clean_piece_ids(gen[0].tolist(), bos_id, eos_id, pad_id)
    return sp.decode(hyp_ids)


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    cfg = load_cfg_with_base(args.config)
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    print(f"[INFO] config top-level keys: {list(cfg_plain.keys())}")
    if hasattr(cfg, "_base_path"):
        print(f"[INFO] merged base_config file: {cfg._base_path}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(
            f"Unexpected checkpoint format. Need dict with key 'model'. "
            f"type={type(ckpt)} keys={list(ckpt.keys()) if isinstance(ckpt, dict) else None}"
        )

    sp = load_sentencepiece_from_ckpt(ckpt, ckpt_path=args.ckpt, config_path=args.config)
    pad_id, bos_id, eos_id = sp.pad_id(), sp.bos_id(), sp.eos_id()
    vocab_size = sp.get_piece_size()

    model, max_src_len, max_tgt_len = build_model_from_config(cfg, vocab_size=vocab_size, pad_id=pad_id)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    print(f"[INFO] Loaded ckpt: {args.ckpt}")
    print(f"[INFO] epoch={ckpt.get('epoch')} val_bleu_fast={ckpt.get('val_bleu_fast')}")
    if "sp_model_path" in ckpt:
        print(f"[INFO] sp_model_path={ckpt.get('sp_model_path')}")
    print(f"[INFO] vocab_size={vocab_size} pad/bos/eos={pad_id}/{bos_id}/{eos_id}")
    print(f"[INFO] max_src_len={max_src_len} max_tgt_len={max_tgt_len}")

    if not args.cli:
        raise RuntimeError("This script supports CLI only. Please add --cli.")

    print("Enter English (empty line to quit):")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        zh = translate_one(
            model=model,
            sp=sp,
            text=line,
            device=device,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            decode=args.decode,
            beam_size=args.beam_size,
            len_penalty=args.len_penalty,
            max_len=args.max_len,
        )
        print(zh)


if __name__ == "__main__":
    main()