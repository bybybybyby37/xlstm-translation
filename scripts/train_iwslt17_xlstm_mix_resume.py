'''
Command:
python -m scripts.train_iwslt17_xlstm_mix_resume \
  --config config/iwslt17_xlstm11_mix10to1_resume.yaml \
  --variant 11 \
  --eval_split test
'''
import os
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import sentencepiece as spm

import argparse
from dataclasses import replace
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig

from src.iwslt17_data import set_seed
from src.mixed_iwslt17_data import create_mixed_iwslt17_dataloaders
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config, e.g. config/iwslt17_xlstm11_mix10to1_resume.yaml",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="11",
        help="Just for logging & ckpt naming: 01 / 10 / 11",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to use for final BLEU/loss evaluation.",
    )
    return parser.parse_args()


def run_eval(model, dataloader, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            logits = model(src, tgt_in)  # [B,T,V]
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


def load_and_merge_config(path: str):
    cfg = OmegaConf.load(path)
    if "base_config" in cfg and cfg.base_config:
        base = OmegaConf.load(cfg.base_config)
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def train_mix_resume(args):
    # ----- load config (supports base_config merge) -----
    cfg = load_and_merge_config(args.config)

    training_cfg = cfg.training
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    # ----- seed -----
    seed = int(training_cfg.get("seed", 1337))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}, seed={seed}")

    # ----- hyperparameters -----
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
    min_delta = float(training_cfg.get("min_delta", 0.0))

    # ----- mixed data -----
    spm_model_path = str(dataset_cfg.get("spm_model_path", "spm/iwslt17_en_zh_8000.model"))
    synthetic_txt_path = str(dataset_cfg.get("synthetic_txt_path", "data/synth_kept.en_zh.txt"))
    ratio_iwslt = int(dataset_cfg.get("mix_ratio_iwslt", 10))
    ratio_synth = int(dataset_cfg.get("mix_ratio_synth", 1))

    sp, train_loader, val_loader, test_loader = create_mixed_iwslt17_dataloaders(
        spm_model_path=spm_model_path,
        synthetic_txt_path=synthetic_txt_path,
        vocab_size=vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        ratio_iwslt=ratio_iwslt,
        ratio_synth=ratio_synth,
    )

    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # ----- model config from yaml -----
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] parameters: {n_params/1e6:.2f} M")

    # ----- resume from original best ckpt (model weights) -----
    resume_ckpt = str(training_cfg.get("resume_ckpt", "checkpoints/xlstm_iwslt17_en_zh_11.pt"))
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[Resume] loaded model weights from: {resume_ckpt} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

    # optimizer/scheduler re-init (original ckpt did not store optimizer states)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    run_dir = f"runs/iwslt17_xlstm_{args.variant}_mix{ratio_iwslt}to{ratio_synth}_resume_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    out_ckpt_name = f"xlstm_iwslt17_en_zh_{args.variant}_mix{ratio_iwslt}to{ratio_synth}_resume.pt"
    ckpt_path = os.path.join("checkpoints", out_ckpt_name)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0

    print("[Train] start training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            ncols=100,
            leave=False,
        )

        for src, tgt_in, tgt_out in progress:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(src, tgt_in)
            B, T, V = logits.shape

            loss_sum = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
                ignore_index=pad_id,
                reduction="sum",
            )

            ntok = (tgt_out != pad_id).sum().item()
            loss = loss_sum / max(1, ntok)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss_sum += loss_sum.item()
            epoch_tokens += ntok

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss_sum / max(1, epoch_tokens)
        val_loss = run_eval(model, val_loader, device, pad_id)

        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch}: train={avg_train:.4f}, val={val_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "sp_model_path": spm_model_path,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "resume_from": resume_ckpt,
                    "mix_ratio": f"{ratio_iwslt}:{ratio_synth}",
                    "seed": seed,
                },
                ckpt_path,
            )
            print(f"  >> improved, saved checkpoint at {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  >> no improvement ({epochs_no_improve}/{patience_epochs})")
            if epochs_no_improve >= patience_epochs:
                print("[Train] early stopping.")
                break

    # ----- Evaluate best ckpt on eval split -----
    print(f"[Eval] evaluating best checkpoint on {args.eval_split}...")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model"], strict=True)
    model.to(device)
    model.eval()

    # use file-based spm to ensure consistent decode
    sp_eval = spm.SentencePieceProcessor()
    sp_eval.load(spm_model_path)

    pad_id = sp_eval.pad_id()
    bos_id = sp_eval.bos_id()
    eos_id = sp_eval.eos_id()

    eval_loader = val_loader if args.eval_split == "val" else test_loader
    eval_loss_value = run_eval(model, eval_loader, device, pad_id)
    print(f"{args.eval_split.upper()} loss: {eval_loss_value:.4f}, ppl={math.exp(eval_loss_value):.2f}")

    bleu_metric = BLEU(tokenize="zh")
    refs = []
    hyps = []

    with torch.no_grad():
        for src, tgt_in, tgt_out in eval_loader:
            for i in range(src.size(0)):
                src_i = src[i : i + 1].to(device)
                enc_out, src_mask = model.encode(src_i)

                gen_ids = model.beam_decode(
                    enc_out,
                    src_mask,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=max_tgt_len,
                    beam_size=4,
                    len_penalty=0.6,
                )

                hyp_ids = clean_piece_ids(gen_ids[0].tolist(), bos_id, eos_id, pad_id)
                ref_ids = clean_piece_ids(tgt_out[i].tolist(), bos_id, eos_id, pad_id)

                hyps.append(sp_eval.decode(hyp_ids))
                refs.append(sp_eval.decode(ref_ids))

    bleu = bleu_metric.corpus_score(hyps, [refs])
    print("BLEU signature:", bleu_metric.get_signature())
    print(f"{args.eval_split.upper()} BLEU: {bleu.score:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_mix_resume(args)
