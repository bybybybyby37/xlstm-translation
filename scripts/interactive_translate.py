#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import replace

import torch
import sentencepiece as spm
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

from xlstm import xLSTMBlockStackConfig
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    p = argparse.ArgumentParser("Interactive EN->ZH translator (xLSTM IWSLT17)")
    p.add_argument("--config", type=str, default="config/iwslt17_xlstm11.yaml", help="YAML config path")
    p.add_argument("--ckpt", type=str, default="checkpoints/xlstm_iwslt17_en_zh_11.pt", help="Checkpoint .pt path")
    p.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    p.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"], help="Decoding method")
    p.add_argument("--beam_size", type=int, default=4, help="Beam size if decode=beam")
    p.add_argument("--len_penalty", type=float, default=0.6, help="Length penalty for beam search")
    p.add_argument("--max_len", type=int, default=None, help="Max target length (default from config)")
    p.add_argument("--port", type=int, default=7860, help="Gradio server port")
    p.add_argument("--share", action="store_true", help="Create a public share link (if allowed)")
    p.add_argument("--cli", action="store_true", help="Run in terminal (no UI)")
    return p.parse_args()


def clean_piece_ids(ids, bos_id, eos_id, pad_id):
    """Remove PAD, truncate at EOS, and drop BOS before decoding."""
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


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_from_config(cfg, vocab_size: int, pad_id: int):
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    max_src_len = int(dataset_cfg.max_src_len)
    max_tgt_len = int(dataset_cfg.max_tgt_len)

    # xLSTM block stack config from yaml
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    base_stack_cfg = from_dict(
        data_class=xLSTMBlockStackConfig,
        data=model_cfg_dict,
        config=DaciteConfig(strict=True),
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
    decode: str = "beam",
    beam_size: int = 4,
    len_penalty: float = 0.6,
    max_len: int | None = None,
) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # [BOS] + pieces + [EOS]
    src_ids = [bos_id] + sp.encode(text, out_type=int) + [eos_id]

    # truncate/pad to max_src_len
    src_ids = src_ids[:max_src_len]
    if len(src_ids) < max_src_len:
        src_ids = src_ids + [pad_id] * (max_src_len - len(src_ids))

    src = torch.tensor([src_ids], dtype=torch.long, device=device)  # [1, S]
    enc_out, src_mask = model.encode(src)

    # decode
    use_max_len = int(max_len) if (max_len is not None) else int(max_tgt_len)
    if decode == "beam":
        gen = model.beam_decode(
            enc_out, src_mask,
            bos_id=bos_id, eos_id=eos_id,
            max_len=use_max_len,
            beam_size=beam_size,
            len_penalty=len_penalty,
        )
    else:
        gen = model.greedy_decode(
            enc_out, src_mask,
            bos_id=bos_id, eos_id=eos_id,
            max_len=use_max_len,
        )

    hyp_ids = clean_piece_ids(gen[0].tolist(), bos_id, eos_id, pad_id)
    return sp.decode(hyp_ids)


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # load config
    cfg = OmegaConf.load(args.config)

    # load checkpoint (expects keys: model, sp_model, epoch, val_loss)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "sp_model" not in ckpt:
        raise RuntimeError(
            "Unexpected checkpoint format. Expected a dict with keys: 'model' and 'sp_model'."
        )

    # load sentencepiece from serialized proto in ckpt
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(ckpt["sp_model"])
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()

    # build model and load weights
    model, max_src_len, max_tgt_len = build_model_from_config(cfg, vocab_size=vocab_size, pad_id=pad_id)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"[INFO] Loaded ckpt: {args.ckpt}")
    print(f"[INFO] epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")
    print(f"[INFO] sp vocab_size={vocab_size} pad/bos/eos={pad_id}/{bos_id}/{eos_id}")
    print(f"[INFO] max_src_len={max_src_len} max_tgt_len={max_tgt_len}")

    def _run_once(x: str, decode: str, beam_size: int, len_penalty: float, max_len: int):
        return translate_one(
            model=model,
            sp=sp,
            text=x,
            device=device,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            decode=decode,
            beam_size=beam_size,
            len_penalty=len_penalty,
            max_len=max_len,
        )

    # CLI mode
    if args.cli:
        print("Enter English (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            zh = _run_once(
                line,
                args.decode,
                args.beam_size,
                args.len_penalty,
                int(args.max_len) if args.max_len is not None else int(max_tgt_len),
            )
            print(zh)
        return

    # UI mode (Gradio)
    try:
        import gradio as gr
    except Exception as e:
        raise RuntimeError("Gradio not installed. Install with: pip install gradio") from e

    default_max_len = int(args.max_len) if args.max_len is not None else int(max_tgt_len)

    with gr.Blocks(title="xLSTM IWSLT17 EN→ZH Translator") as demo:
        gr.Markdown("## xLSTM IWSLT17 EN→ZH Translator\nInput in English, and output the Chinese translation of the checkpoint.\n输入英文，输出该 checkpoint 的中文翻译结果。")

        inp = gr.Textbox(label="English", lines=3, placeholder="Type an English sentence here...")
        out = gr.Textbox(label="Chinese", lines=4)

        with gr.Row():
            decode = gr.Dropdown(choices=["beam", "greedy"], value=args.decode, label="Decode")
            beam = gr.Slider(1, 8, value=args.beam_size, step=1, label="Beam size (beam only)")
        with gr.Row():
            lp = gr.Slider(0.0, 2.0, value=args.len_penalty, step=0.05, label="Length penalty (beam only)")
            ml = gr.Slider(8, int(max_tgt_len), value=default_max_len, step=1, label="Max output length")

        btn = gr.Button("Translate")

        btn.click(fn=_run_once, inputs=[inp, decode, beam, lp, ml], outputs=[out])
        inp.submit(fn=_run_once, inputs=[inp, decode, beam, lp, ml], outputs=[out])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
