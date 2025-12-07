import os
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sacrebleu import corpus_bleu

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
    return parser.parse_args()

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
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    run_dir = f"runs/iwslt17_xlstm_{args.variant}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"xlstm_iwslt17_en_zh_{args.variant}.pt")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0

    def run_eval(dataloader):
        model.eval()
        losses = []
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
                )
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print("Start training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
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

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
                ignore_index=pad_id,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)
        val_loss = run_eval(val_loader)
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

    # ---- Cal loss on test dataset + a simply BLEU ----
    print("Evaluating best checkpoint on test set...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    def eval_loss(loader):
        return run_eval(loader)

    test_loss = eval_loss(test_loader)
    print(f"Test loss: {test_loss:.4f}, ppl={math.exp(test_loss):.2f}")

    # Use greedy decode to cal BLEU on first N batches on test dataset
    N = 200
    refs = []
    hyps = []

    with torch.no_grad():
        count = 0
        for src, tgt_in, tgt_out in test_loader:
            # Here simply batch=1 for BLEU
            for i in range(src.size(0)):
                src_i = src[i : i + 1].to(device)
                tgt_ids = tgt_in[i : i + 1].to(device)

                enc_out, src_mask = model.encode(src_i)
                gen_ids = model.greedy_decode(
                    enc_out, src_mask, bos_id=bos_id, eos_id=eos_id, max_len=max_tgt_len
                )  # [1, L]

                # Decode into a string
                hyp = sp.decode(gen_ids[0].tolist())
                ref = sp.decode(tgt_out[i].tolist())

                hyps.append(hyp)
                refs.append([ref])  # sacrebleu needs list of list

                count += 1
                if count >= N:
                    break
            if count >= N:
                break

    bleu = corpus_bleu(hyps, refs)
    print(f"Test BLEU on {N} samples: {bleu.score:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_iwslt17_xlstm(args)
