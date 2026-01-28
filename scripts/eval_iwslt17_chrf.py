'''
Command:
python -m scripts.eval_iwslt17_chrf \
  --config config/iwslt17_xlstm11.yaml \
  --ckpt checkpoints/xlstm_iwslt17_en_zh_11.pt \
  --split test \
  --max_sentences -1 \
  --beam_size 4 --max_len 128 --len_penalty 0.6 \
  --num_workers 0\
  --chrf_word_order 0

python -m scripts.eval_iwslt17_chrf \
  --config config/iwslt17_xlstm11_mix10to1_anchor_synth.yaml \
  --ckpt checkpoints/xlstm_iwslt17_en_zh_11_anchorS_mix10to1_bestbleu_currentBEST.pt \
  --split test \
  --max_sentences -1 \
  --beam_size 4 --max_len 128 --len_penalty 0.6 \
  --num_workers 0 \
  --chrf_word_order 0

'''
import os
import math
import argparse
from dataclasses import replace

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import sentencepiece as spm
from sacrebleu.metrics import CHRF

from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig

from src.iwslt17_data import set_seed, collate_translation_batch, IWSLT17EnZhDataset
from src.xlstm_seq2seq import XlstmSeq2Seq


def load_and_merge_config(path: str):
    cfg = OmegaConf.load(path)
    if "base_config" in cfg and cfg.base_config:
        base = OmegaConf.load(cfg.base_config)
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def clean_piece_ids(ids, bos_id, eos_id, pad_id):
    out = []
    for x in ids:
        if x == pad_id:
            continue
        if x == eos_id:
            break
        if x == bos_id:
            continue
        out.append(x)
    return out


def run_loss(model, dataloader, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tok = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            logits = model(src, tgt_in)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
                ignore_index=pad_id,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tok += (tgt_out != pad_id).sum().item()
    return total_loss / max(1, total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml config (can include base_config)")
    ap.add_argument("--ckpt", required=True, help="checkpoint path to evaluate")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max_sentences", type=int, default=-1, help="-1 = full split")
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--len_penalty", type=float, default=0.6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chrf_word_order", type=int, default=0, help="ChrF word order (default 0)")
    args = ap.parse_args()

    cfg = load_and_merge_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}, seed={args.seed}")

    # SentencePiece
    spm_model_path = str(cfg.dataset.get("spm_model_path", "spm/iwslt17_en_zh_8000.model"))
    if not os.path.exists(spm_model_path):
        raise FileNotFoundError(spm_model_path)
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)

    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # Dataset
    dataset_dict = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)
    split_name = "validation" if args.split == "val" else "test"
    ds = IWSLT17EnZhDataset(
        dataset_dict[split_name],
        sp,
        int(cfg.dataset.max_src_len),
        int(cfg.dataset.max_tgt_len),
        "en",
        "zh",
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_translation_batch(b, pad_id=pad_id),
        drop_last=False,
    )

    # Build model
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    base_stack_cfg = from_dict(
        data_class=xLSTMBlockStackConfig,
        data=model_cfg_dict,
        config=DaciteConfig(strict=True),
    )
    enc_cfg = replace(base_stack_cfg, context_length=int(cfg.dataset.max_src_len))
    dec_cfg = replace(base_stack_cfg, context_length=int(cfg.dataset.max_tgt_len))

    model = XlstmSeq2Seq(
        vocab_size=int(cfg.dataset.vocab_size),
        max_src_len=int(cfg.dataset.max_src_len),
        max_tgt_len=int(cfg.dataset.max_tgt_len),
        pad_id=pad_id,
        enc_cfg=enc_cfg,
        dec_cfg=dec_cfg,
    ).to(device)

    # Load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Compatible with your saved format: {"model": state_dict, ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # fallback: allow pure state_dict checkpoints
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    print(f"[Load] ckpt={args.ckpt}")

    # (Optional) also report loss/ppl for reference
    loss = run_loss(model, dl, device, pad_id)
    print(f"{args.split.upper()} loss: {loss:.4f}, ppl={math.exp(loss):.2f}")

    # Decode + ChrF
    chrf_metric = CHRF(word_order=args.chrf_word_order)
    refs, hyps = [], []
    done = 0

    with torch.no_grad():
        for src, tgt_in, tgt_out in tqdm(dl, desc="ChrF decoding", ncols=100):
            bs = src.size(0)
            for i in range(bs):
                src_i = src[i : i + 1].to(device)
                enc_out, src_mask = model.encode(src_i)

                gen_ids = model.beam_decode(
                    enc_out,
                    src_mask,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=args.max_len,
                    beam_size=args.beam_size,
                    len_penalty=args.len_penalty,
                )

                hyp_ids = clean_piece_ids(gen_ids[0].tolist(), bos_id, eos_id, pad_id)
                ref_ids = clean_piece_ids(tgt_out[i].tolist(), bos_id, eos_id, pad_id)

                hyps.append(sp.decode(hyp_ids))
                refs.append(sp.decode(ref_ids))

                done += 1
                if args.max_sentences > 0 and done >= args.max_sentences:
                    break
            if args.max_sentences > 0 and done >= args.max_sentences:
                break

    score = chrf_metric.corpus_score(hyps, [refs]).score
    print(f"{args.split.upper()} ChrF: {score:.2f}  (decoded {done} sentences, word_order={args.chrf_word_order})")


if __name__ == "__main__":
    main()
