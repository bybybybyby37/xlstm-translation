'''
Command:
python -m scripts.train_iwslt17_xlstm_mix_resume_anchor_synth \
  --config config/iwslt17_xlstm11_mix10to1_anchor_synth.yaml \
  --variant 11 \
  --eval_split test

'''
import os
import time
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import sentencepiece as spm

from dataclasses import replace
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig

from src.iwslt17_data import set_seed
from src.mixed_iwslt17_data_anchor_synth import (
    prepare_iwslt17_and_synth,
    MixedAnchorSynthDataset,
)
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--variant", default="11", type=str)
    p.add_argument("--eval_split", default="test", choices=["val", "test"])
    return p.parse_args()


def load_and_merge_config(path: str):
    cfg = OmegaConf.load(path)
    if "base_config" in cfg and cfg.base_config:
        base = OmegaConf.load(cfg.base_config)
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def run_eval_loss(model, dataloader, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
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
            total_tokens += (tgt_out != pad_id).sum().item()
    model.train()
    return total_loss / max(1, total_tokens)


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


def run_eval_bleu_fast(
    *,
    model,
    dataloader,
    device,
    sp: spm.SentencePieceProcessor,
    max_sentences: int,
    beam_size: int,
    max_len: int,
    len_penalty: float,
    tokenize: str = "zh",
):
    """
    Fast BLEU on validation: decode only first max_sentences sentences (deterministic order).
    """
    model.eval()

    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    bleu_metric = BLEU(tokenize=tokenize)
    refs = []
    hyps = []

    done = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            bs = src.size(0)
            for i in range(bs):
                src_i = src[i : i + 1].to(device)
                enc_out, src_mask = model.encode(src_i)

                gen_ids = model.beam_decode(
                    enc_out,
                    src_mask,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=max_len,
                    beam_size=beam_size,
                    len_penalty=len_penalty,
                )

                hyp_ids = clean_piece_ids(gen_ids[0].tolist(), bos_id, eos_id, pad_id)
                ref_ids = clean_piece_ids(tgt_out[i].tolist(), bos_id, eos_id, pad_id)

                hyps.append(sp.decode(hyp_ids))
                refs.append(sp.decode(ref_ids))

                done += 1
                if max_sentences > 0 and done >= max_sentences:
                    break
            if max_sentences > 0 and done >= max_sentences:
                break

    bleu = bleu_metric.corpus_score(hyps, [refs]).score
    model.train()
    return bleu, done


def train(args):
    cfg = load_and_merge_config(args.config)
    training_cfg = cfg.training
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    seed = int(training_cfg.get("seed", 1337))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}, seed={seed}")

    vocab_size = int(dataset_cfg.vocab_size)
    max_src_len = int(dataset_cfg.max_src_len)
    max_tgt_len = int(dataset_cfg.max_tgt_len)

    batch_size = int(training_cfg.batch_size)
    num_epochs = int(training_cfg.num_epochs)
    lr = float(training_cfg.lr)
    weight_decay = float(training_cfg.weight_decay)
    grad_clip = float(training_cfg.grad_clip)
    patience_epochs = int(training_cfg.patience_epochs)
    num_workers = int(training_cfg.num_workers)
    min_delta_bleu = float(training_cfg.get("min_delta_bleu", 0.0))  # optional

    # fast val BLEU settings
    val_bleu_max_sent = int(training_cfg.get("val_bleu_max_sentences", 500))
    val_bleu_beam = int(training_cfg.get("val_bleu_beam_size", 1))
    val_bleu_max_len = int(training_cfg.get("val_bleu_max_len", 80))
    val_bleu_len_pen = float(training_cfg.get("val_bleu_len_penalty", 0.6))
    val_bleu_tok = str(training_cfg.get("val_bleu_tokenize", "zh"))

    spm_model_path = str(dataset_cfg.get("spm_model_path", "spm/iwslt17_en_zh_8000.model"))
    synthetic_txt_path = str(dataset_cfg.get("synthetic_txt_path", "data/synth_kept.en_zh.txt"))
    ratio_iwslt = int(dataset_cfg.get("mix_ratio_iwslt", 10))

    sp, iwslt_train, synth_set, val_loader, test_loader, collate_fn = prepare_iwslt17_and_synth(
        spm_model_path=spm_model_path,
        synthetic_txt_path=synthetic_txt_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    pad_id = sp.pad_id()

    # model config from yaml
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
    ).to(device)

    # resume
    resume_ckpt = str(training_cfg.get("resume_ckpt", "checkpoints/xlstm_iwslt17_en_zh_11.pt"))
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(resume_ckpt)
    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[Resume] loaded model from: {resume_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Keep scheduler on val_loss (stable). Early stop is driven by BLEU.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    run_dir = f"runs/iwslt17_xlstm_{args.variant}_anchorS_mix{ratio_iwslt}to1_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_best_path = f"checkpoints/xlstm_iwslt17_en_zh_{args.variant}_anchorS_mix{ratio_iwslt}to1_bestbleu.pt"

    best_bleu = -1e9
    bad = 0
    global_step = 0

    print("[Train] start training... (Anchor on synthetic: use ALL synth, sample IWSLT=ratio*synth)")
    for epoch in range(1, num_epochs + 1):
        mixed_train = MixedAnchorSynthDataset(
            iwslt_train=iwslt_train,
            synth=synth_set,
            ratio_iwslt=ratio_iwslt,
            seed=seed,
            epoch=epoch,
        )
        train_loader = torch.utils.data.DataLoader(
            mixed_train,
            batch_size=batch_size,
            shuffle=False,   # mapping already shuffled deterministically
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

        print(
            f"[Data] Epoch {epoch}: train_size={len(mixed_train)} "
            f"(iwslt_take={mixed_train.n_iwslt_take}, synth={len(synth_set)}), ratio={ratio_iwslt}:1"
        )

        model.train()
        loss_sum = 0.0
        tok_sum = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100, leave=False)
        for src, tgt_in, tgt_out in prog:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt_in)
            B, T, V = logits.shape

            ce_sum = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
                ignore_index=pad_id,
                reduction="sum",
            )
            ntok = (tgt_out != pad_id).sum().item()
            loss = ce_sum / max(1, ntok)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            loss_sum += ce_sum.item()
            tok_sum += ntok

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            prog.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = loss_sum / max(1, tok_sum)
        val_loss = run_eval_loss(model, val_loader, device, pad_id)
        val_bleu_fast, used = run_eval_bleu_fast(
            model=model,
            dataloader=val_loader,
            device=device,
            sp=sp,
            max_sentences=val_bleu_max_sent,
            beam_size=val_bleu_beam,
            max_len=val_bleu_max_len,
            len_penalty=val_bleu_len_pen,
            tokenize=val_bleu_tok,
        )

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/bleu_fast", val_bleu_fast, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch}: train={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_bleu_fast={val_bleu_fast:.2f} (n={used}), lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        scheduler.step(val_loss)

        # ---- early stop / best ckpt based on val_bleu_fast (maximize) ----
        if val_bleu_fast > best_bleu + min_delta_bleu:
            best_bleu = val_bleu_fast
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "sp_model_path": spm_model_path,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_bleu_fast": val_bleu_fast,
                    "val_bleu_fast_n": used,
                    "resume_from": resume_ckpt,
                    "mix_anchor": "synthetic",
                    "mix_ratio": f"{ratio_iwslt}:1",
                    "seed": seed,
                    "bleu_fast_cfg": {
                        "max_sentences": val_bleu_max_sent,
                        "beam_size": val_bleu_beam,
                        "max_len": val_bleu_max_len,
                        "len_penalty": val_bleu_len_pen,
                        "tokenize": val_bleu_tok,
                    },
                },
                ckpt_best_path,
            )
            print(f"  >> improved BLEU, saved best checkpoint at {ckpt_best_path}")
        else:
            bad += 1
            print(f"  >> no BLEU improvement ({bad}/{patience_epochs})")
            if bad >= patience_epochs:
                print("[Train] early stopping (by val_bleu_fast).")
                break

    print(f"[Done] best ckpt (by val_bleu_fast): {ckpt_best_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
