import os
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sacrebleu import corpus_bleu
from sacrebleu.metrics import BLEU
import sentencepiece as spm

import argparse
from dataclasses import replace
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig

from src.iwslt17_data import (
    create_iwslt17_dataloaders,
    set_seed,
)
from src.xlstm_seq2seq import XlstmSeq2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to xLSTM yaml config, e.g. config/iwslt17_xlstm10.yaml",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="10",
        help="Just for logging: 01 / 10 / 11",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to use for final BLEU/loss evaluation (use val for tuning, test for final).",
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

            # token-sum loss (more correct than averaging per batch)
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
    """
    Remove PAD, truncate at EOS, and drop BOS/EOS before SentencePiece decode.
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

def train_iwslt17_xlstm(args):
    # ----- load config -----
    cfg = OmegaConf.load(args.config)

    training_cfg = cfg.training
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    # ----- seed -----
    set_seed(int(training_cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- hyperparameters from yaml -----
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

    # ---- data & dataloader ----
    sp, train_loader, val_loader, test_loader = create_iwslt17_dataloaders(
        vocab_size=vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=int(training_cfg.seed),
    )
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # ---- xLSTM block stack config from yaml ----
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

    base_stack_cfg = from_dict(
        data_class=xLSTMBlockStackConfig,
        data=model_cfg_dict,
        config=DaciteConfig(strict=True),
    )

    # encoder / decoder share the same block config, 
    # but can be with different context_length (src vs tgt)
    enc_cfg = replace(base_stack_cfg, context_length=max_src_len)
    dec_cfg = replace(base_stack_cfg, context_length=max_tgt_len)

    # ---- model ----
    model = XlstmSeq2Seq(
        vocab_size=vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        pad_id=pad_id,
        enc_cfg=enc_cfg,
        dec_cfg=dec_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    run_dir = f"runs/iwslt17_xlstm_{args.variant}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"xlstm_iwslt17_en_zh_{args.variant}.pt")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0

    print("Start training...")
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

            # token count (non-pad)
            ntok = (tgt_out != pad_id).sum().item()

            # backprop uses normalized loss (so that lr not affected by batch length)
            loss = loss_sum / max(1, ntok)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss_sum += loss_sum.item()
            epoch_tokens += ntok

            # step log: token-average
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "sp_model": sp.serialized_model_proto(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  >> improved, saved checkpoint at {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  >> no improvement ({epochs_no_improve}/{patience_epochs})")
            if epochs_no_improve >= patience_epochs:
                print("Early stopping.")
                break

    # ---- Evaluate best checkpoint on test set (loss + BLEU) ----
    print("Evaluating best checkpoint on test set...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    sp_ckpt = spm.SentencePieceProcessor()
    sp_ckpt.LoadFromSerializedProto(ckpt["sp_model"])
    sp = sp_ckpt
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    eval_loader = val_loader if args.eval_split == "val" else test_loader
    eval_loss_value = run_eval(model, eval_loader, device, pad_id)
    print(f"{args.eval_split.upper()} loss: {eval_loss_value:.4f}, ppl={math.exp(eval_loss_value):.2f}")

    # ---- BLEU (sacreBLEU), robust + reproducible ----
    # IMPORTANT:
    # 1) Use full test set by default (set N to an int to subsample for quick checks)
    # 2) Strip PAD and truncate at EOS before decoding
    # 3) Use sacreBLEU metric with a fixed tokenizer and print its signature

    bleu_metric = BLEU(tokenize="zh")  # standard, reproducible tokenization

    N = None  # None = full test set; set e.g. 200 for quick sanity checks

    refs = []  # list[str]
    hyps = []  # list[str]

    with torch.no_grad():
        count = 0
        for src, tgt_in, tgt_out in eval_loader:
            for i in range(src.size(0)):
                src_i = src[i : i + 1].to(device)

                enc_out, src_mask = model.encode(src_i)
                gen_ids = model.beam_decode(
                    enc_out, src_mask,
                    bos_id=bos_id, eos_id=eos_id,
                    max_len=max_tgt_len,
                    beam_size=4,
                    len_penalty=0.6,
                )

                hyp_ids = clean_piece_ids(gen_ids[0].tolist(), bos_id, eos_id, pad_id)
                ref_ids = clean_piece_ids(tgt_out[i].tolist(), bos_id, eos_id, pad_id)

                hyp = sp.decode(hyp_ids)
                ref = sp.decode(ref_ids)

                hyps.append(hyp)
                refs.append(ref)

                count += 1
                if (N is not None) and (count >= N):
                    break
            if (N is not None) and (count >= N):
                break

    # sacreBLEU expects: hyps: list[str], refs: list[list[str]] (one list per reference set)
    bleu = bleu_metric.corpus_score(hyps, [refs])
    print("BLEU signature:", bleu_metric.get_signature())
    if N is None:
        print(f"{args.eval_split.upper()} BLEU (full test set): {bleu.score:.2f}")
    else:
        print(f"{args.eval_split.upper()} BLEU on {N} samples: {bleu.score:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_iwslt17_xlstm(args)
