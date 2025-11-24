import sys
import os
import math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# NeurIPS xLSTM LMModel (native, no Large 7B kernels)
from xlstm import (
    xLSTMLMModel,
    xLSTMLMModelConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

def build_vocab_from(text):
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return chars, char2idx, idx2char

class ShakespeareDataset(Dataset):
    def __init__(self, text, char2idx, idx2char, seq_len=128, stride=None):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(self.char2idx)
        self.seq_len = seq_len
        self.data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        self.stride = stride or seq_len
        print(f"Vocabulary size (shared): {self.vocab_size} characters")
        print(f"Total sequence length: {len(self.data)} tokens")

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.stride

    def __getitem__(self, i):
        idx = i * self.stride
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y


def main():
    eval_interval = 100
    learning_rate = 0.002
    weight_decay  = 0.02
    grad_clip     = 1.0
    batch_size = 32
    seq_len = 128
    stride = int(seq_len/2)
    embedding_dim=256
    num_heads=4
    num_blocks=4
    num_epochs = 2
    patience_epochs = 3

    # paths relative to THIS script
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(PROJECT_DIR, "data", "full_shakespeare.txt")
    save_path = os.path.join(PROJECT_DIR, "output", "best.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    seed=1337
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    backend = "vanilla" #"cuda" if torch.cuda.is_available() else "vanilla"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Dataset
    data_path = "data/full_shakespeare.txt"
    if data_path == "data/tiny_shakespeare.txt":
        if os.path.exists(data_path):
            print(f"'{data_path}' already exists, skipping download.")
        else:
            import requests
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url).text
            os.makedirs("data", exist_ok=True)
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(text)
            print("Tiny Shakespeare downloaded! File size:", len(text), "characters")
    elif data_path == "data/full_shakespeare.txt":
        if os.path.exists(data_path):
            print(f"'{data_path}' already exists, skipping download.")
        else:
            import requests
            os.makedirs("data", exist_ok=True)
            url = "https://www.gutenberg.org/files/100/100-0.txt"
            print("Downloading full Shakespeare from Project Gutenberg...")
            text = requests.get(url).text
            if "*** START" in text:
                text = text.split("*** START")[1]
            if "*** END" in text:
                text = text.split("*** END")[0]
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(text)
            print("Full Shakespeare downloaded! File size:", len(text), "characters")
    else:
        print("Unexpected dataset, stop training.")
        sys.exit()

    with open(data_path, "r", encoding='utf-8') as f:
        text_full = f.read()

    n = len(text_full)
    text = text_full[:n]
    train_text = text[:int(0.8*n)]
    val_text   = text[int(0.8*n):int(0.9*n)]
    test_text  = text[int(0.9*n):]

    chars, char2idx, idx2char = build_vocab_from(text)

    train_dataset = ShakespeareDataset(train_text, char2idx, idx2char, seq_len=seq_len, stride=stride)
    val_dataset   = ShakespeareDataset(val_text,   char2idx, idx2char, seq_len=seq_len, stride=stride)
    test_dataset  = ShakespeareDataset(test_text,  char2idx, idx2char, seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    VAL_MAX_BATCHES = 200
    TEST_MAX_BATCHES = 200
    val_num = min(len(val_dataset), VAL_MAX_BATCHES * batch_size)
    test_num = min(len(test_dataset), TEST_MAX_BATCHES * batch_size)

    g = torch.Generator().manual_seed(3407)  # Fixed seed for reproduction
    val_idx = torch.randperm(len(val_dataset), generator=g)[:val_num].tolist()
    test_idx = torch.randperm(len(test_dataset), generator=g)[:test_num].tolist()

    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    val_loader_small  = DataLoader(val_subset,  batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader_small = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)


    config = xLSTMLMModelConfig(
        vocab_size=len(char2idx),
        context_length=seq_len,
        num_blocks=num_blocks,
        embedding_dim=embedding_dim,
        # --- mLSTM block ---
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
            )
        ),
        # --- sLSTM block ---
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",         # In windows we should use vanilla
                num_heads=num_heads,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=1.3,
                act_fn="gelu"
            )
        ),
        # Defines at whitch block index does sLSTM appears, here 1st
        slstm_at=[1],
    )


    model = xLSTMLMModel(config).cuda()
    print("xLSTM model initialized on CUDA")


    run_dir = f"runs/xlstm_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)
    print("TensorBoard logdir:", run_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )


    best_val = float("inf")
    epochs_no_improve = 0
    global_step = 0 

    @torch.no_grad()
    def estimate_loss_small(loader):
        model.eval()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)


    print("Start training on", device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader, 1),
                            total=len(train_loader),
                            desc=f"Epoch {epoch}/{num_epochs}",
                            leave=False, 
                            ncols=100)

        for step, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            writer.add_scalar("loss/train", loss.item(), global_step)

            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})


        val_loss = estimate_loss_small(val_loader_small)
        torch.cuda.empty_cache()
        scheduler.step(val_loss)

        avg_train = epoch_loss / len(train_loader)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("loss/train_epoch_avg", avg_train, epoch)

        print(f"\rEpoch [{epoch}/{num_epochs}] "
            f"┃ train_loss: {avg_train:.4f} "
            f"┃ val_loss: {val_loss:.4f} "
            f"┃ lr: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "epoch": epoch,
                "best_val": best_val,
            }, save_path)
            print(f"Improved! best_val={best_val:.4f} (saved)")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience_epochs})")
            if epochs_no_improve >= patience_epochs:
                print("Early stopping triggered.")
                break


    '''# Test
    @torch.no_grad()
    def evaluate_test_set():
        model.eval()
        losses = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[0]
            loss = F.cross_entropy(logits.view(-1, train_dataset.vocab_size), y.view(-1))
            losses.append(loss.item())
        return sum(losses) / len(losses)'''

    # load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at epoch={checkpoint['epoch']}")

    # test
    test_loss = estimate_loss_small(test_loader_small)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Perplexity (PPL): {math.exp(test_loss):.2f}")


    model.eval()

    encode = lambda s: [train_dataset.char2idx[c] for c in s]
    decode = lambda l: ''.join([train_dataset.idx2char[i] for i in l])

    WINDOW = 64  
    PAD_ID = train_dataset.char2idx.get(' ', 0)

    generated_ids = encode("ROMEO:")

    with torch.inference_mode():
        for step in range(500):
            tail = generated_ids[-WINDOW:]
            if len(tail) < WINDOW:
                x_ids = [PAD_ID] * (WINDOW - len(tail)) + tail
            else:
                x_ids = tail

            x = torch.tensor([x_ids], dtype=torch.long, device=device)

            out = model(x) 
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            topk = 50
            vals, idxs = torch.topk(probs, topk, dim=-1)
            probs = vals / vals.sum(dim=-1, keepdim=True)
            next_id = idxs.gather(-1, torch.multinomial(probs, 1)).item()

            generated_ids.append(next_id)

            if (step + 1) % 50 == 0:
                torch.cuda.empty_cache()

    print("\n=== Generate Text ===")
    print(decode(generated_ids))

if __name__ == "__main__":
    main()