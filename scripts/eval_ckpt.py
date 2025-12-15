import argparse
import math
import os

import torch
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from dataclasses import replace
import sentencepiece as spm

from sacrebleu.metrics import BLEU

from xlstm import xLSTMBlockStackConfig
from src.iwslt17_data import create_iwslt17_dataloaders, set_seed
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    p = argparse.ArgumentParser("Evaluate a saved xLSTM checkpoint on val/test split")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint file")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
    p.add_argument("--max_bleu_samples", type=int, default=-1, help="-1 for full split; otherwise evaluate first N samples")
    p.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"], help="Decoding method for BLEU")
    p.add_argument("--beam_size", type=int, default=4, help="Beam size if decode=beam")
    p.add_argument("--len_penalty", type=float, default=0.6, help="Length penalty for beam search")
    return p.parse_args()


def clean_piece_ids(ids, bos_id, eos_id, pad_id):
    """
    Remove PAD, truncate at EOS, and drop BOS before decoding.
    """
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


@torch.no_grad()
def compute_loss(model, loader, device, pad_id):
    """
    Computes average token-level cross-entropy loss over a loader.
    Assumes loader yields (src, tgt_in, tgt_out) where tgt_out is target tokens aligned with logits.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    ce = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    for src, tgt_in, tgt_out in loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits = model(src, tgt_in)  # [B,T,V]
        B, T, V = logits.shape

        loss = ce(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        total_loss += loss.item()

        # count non-pad target tokens
        total_tokens += (tgt_out != pad_id).sum().item()

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def compute_bleu(model, loader, device, sp, bos_id, eos_id, pad_id, max_tgt_len,
                 decode="beam", beam_size=4, len_penalty=0.6, max_samples=-1):
    """
    Computes sacreBLEU on (hyp, ref) pairs.
    Uses full split by default unless max_samples > 0.
    """
    model.eval()
    bleu_metric = BLEU(tokenize="zh")

    lengths = []  # store len(hyp_ids) after cleaning
    printed = 0   # how many examples printed
    print_samples = 5
    refs = []
    hyps = []
    count = 0

    for src, tgt_in, tgt_out in loader:
        for i in range(src.size(0)):
            src_i = src[i:i+1].to(device)

            enc_out, src_mask = model.encode(src_i)

            # Decode
            if decode == "beam" and hasattr(model, "beam_decode"):
                gen_ids = model.beam_decode(
                    enc_out, src_mask,
                    bos_id=bos_id, eos_id=eos_id,
                    max_len=max_tgt_len,
                    beam_size=beam_size,
                    len_penalty=len_penalty,
                )
            else:
                # fallback to greedy
                gen_ids = model.greedy_decode(
                    enc_out, src_mask,
                    bos_id=bos_id, eos_id=eos_id,
                    max_len=max_tgt_len,
                )

            hyp_ids = clean_piece_ids(gen_ids[0].tolist(), bos_id, eos_id, pad_id)
            ref_ids = clean_piece_ids(tgt_out[i].tolist(), bos_id, eos_id, pad_id)

            hyp = sp.decode(hyp_ids)
            ref = sp.decode(ref_ids)

            hyps.append(hyp)
            refs.append(ref)
            lengths.append(len(hyp_ids))
            if printed < print_samples:
                # also decode source for inspection
                src_ids = clean_piece_ids(src_i[0].tolist(), bos_id, eos_id, pad_id)
                src_text = sp.decode(src_ids)

                print("------------------------------------------------------------")
                print(f"[EXAMPLE {printed+1}] hyp_len={len(hyp_ids)}")
                print("SRC:", src_text)
                print("REF:", ref)
                print("HYP:", hyp)
                printed += 1

            count += 1
            if max_samples > 0 and count >= max_samples:
                break
        if max_samples > 0 and count >= max_samples:
            break

    bleu = bleu_metric.corpus_score(hyps, [refs])
    print("BLEU signature:", bleu_metric.get_signature())
    
    if len(lengths) > 0:
        lengths_sorted = sorted(lengths)
        avg_len = sum(lengths) / len(lengths)
        p50 = lengths_sorted[len(lengths_sorted) // 2]
        zero_ratio = sum(1 for x in lengths if x == 0) / len(lengths)
        le2_ratio = sum(1 for x in lengths if x <= 2) / len(lengths)

        print("============================================================")
        print(f"[LEN STATS] n={len(lengths)} | avg={avg_len:.2f} | p50={p50} | "
            f"ratio(len=0)={zero_ratio:.3f} | ratio(len<=2)={le2_ratio:.3f}")
        print("============================================================")
    return bleu.score, count


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    cfg = OmegaConf.load(args.config)
    training_cfg = cfg.training
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    seed = int(training_cfg.seed)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # data
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

    # model config
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

    # build model (vocab_size required in your project)
    model = XlstmSeq2Seq(vocab_size=vocab_size, enc_cfg=enc_cfg, dec_cfg=dec_cfg).to(device)

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(ckpt["sp_model"])
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()

    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    e = ckpt["epoch"]
    v = ckpt["val_loss"]
    print(f"[INFO] Epoch: {e}")
    print(f"[INFO] val loss: {v}")

    # pick split
    loader = val_loader if args.split == "val" else test_loader

    # loss/ppl
    loss = compute_loss(model, loader, device, pad_id)
    ppl = math.exp(loss)
    print(f"[RESULT] {args.split.upper()} loss={loss:.4f}, ppl={ppl:.2f}")

    # BLEU
    bleu, used = compute_bleu(
        model=model,
        loader=loader,
        device=device,
        sp=sp,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_tgt_len=max_tgt_len,
        decode=args.decode,
        beam_size=args.beam_size,
        len_penalty=args.len_penalty,
        max_samples=args.max_bleu_samples,
    )
    mode = args.decode
    if args.decode == "beam" and not hasattr(model, "beam_decode"):
        mode = "greedy (beam_decode not found; fallback)"
    if args.max_bleu_samples > 0:
        print(f"[RESULT] {args.split.upper()} BLEU ({mode}) on {used} samples: {bleu:.2f}")
    else:
        print(f"[RESULT] {args.split.upper()} BLEU ({mode}) on full split ({used} samples): {bleu:.2f}")


if __name__ == "__main__":
    main()
