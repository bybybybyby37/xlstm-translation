import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)


class XlstmSeq2Seq(nn.Module):
    """
    simplified xLSTM encoder-decoder: 
      - Encoder: xLSTMBlockStack (configurable)
      - Decoder: xLSTMBlockStack (configurable)
      - Cross-attention: torch.nn.MultiheadAttention
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_blocks_enc: int = 4,
        num_blocks_dec: int = 4,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        pad_id: int = 0,
        enc_cfg: xLSTMBlockStackConfig | None = None,  # NEW
        dec_cfg: xLSTMBlockStackConfig | None = None,  # NEW
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # If enc_cfg / dec_cfg are passed，use embedding_dim from config，
        # elsewise use default embedding_dim value
        if enc_cfg is not None:
            embedding_dim = enc_cfg.embedding_dim
        elif dec_cfg is not None:
            embedding_dim = dec_cfg.embedding_dim
        self.embedding_dim = embedding_dim

        # A shared embedding is also possible, here seperate for src/tgt differences
        self.src_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # ----- Encoder config -----
        if enc_cfg is None:
            enc_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4,
                        num_heads=num_heads,
                    )
                ),
                context_length=max_src_len,
                num_blocks=num_blocks_enc,
                embedding_dim=embedding_dim,
            )
        else:
            # To ensure context_lengthis same as actual max-length from src
            enc_cfg = replace(enc_cfg, context_length=max_src_len)

        self.encoder = xLSTMBlockStack(enc_cfg)

        # ----- Decoder config -----
        if dec_cfg is None:
            dec_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4,
                        num_heads=num_heads,
                    )
                ),
                context_length=max_tgt_len,
                num_blocks=num_blocks_dec,
                embedding_dim=embedding_dim,
            )
        else:
            dec_cfg = replace(dec_cfg, context_length=max_tgt_len)

        self.decoder = xLSTMBlockStack(dec_cfg)

        # Cross-attention: decoder hidden attends encoder outputs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=False
        )

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    # --------- forward for Training ---------

    def forward(self, src_ids, tgt_in_ids):
        """
        src_ids:   [B, S]
        tgt_in_ids:[B, T]
        return logits: [B, T, V]
        """
        B, S = src_ids.shape
        _, T = tgt_in_ids.shape

        src_mask = src_ids.eq(self.pad_id)  # [B, S]

        # Encoder
        enc_x = self.src_embed(src_ids)   # [B, S, E]
        enc_out = self.encoder(enc_x)     # [B, S, E]

        # Decoder
        dec_x = self.tgt_embed(tgt_in_ids)  # [B, T, E]
        dec_h = self.decoder(dec_x)         # [B, T, E]

        # Cross-attention: q: [T,B,E], k/v: [S,B,E]
        q = dec_h.transpose(0, 1)
        k = enc_out.transpose(0, 1)
        v = enc_out.transpose(0, 1)

        attn_out, _ = self.cross_attn(
            q, k, v, key_padding_mask=src_mask
        )  # [T,B,E]

        dec_ctx = dec_h + attn_out.transpose(0, 1)  # residual, [B, T, E]
        logits = self.output_proj(dec_ctx)          # [B, T, V]
        return logits

    # --------- encode/decode_step for Inference ---------

    def encode(self, src_ids):
        """
        src_ids: [B, S]
        return (enc_out, src_mask)
        """
        src_mask = src_ids.eq(self.pad_id)
        enc_x = self.src_embed(src_ids)
        enc_out = self.encoder(enc_x)
        return enc_out, src_mask

    def decode_step(self, enc_out, src_mask, tgt_ids):
        """
        enc_out: [B, S, E]
        src_mask: [B, S]
        tgt_ids: [B, T] (currently generated seq)
        return logits: [B, T, V]
        """
        dec_x = self.tgt_embed(tgt_ids)
        dec_h = self.decoder(dec_x)
        q = dec_h.transpose(0, 1)
        k = enc_out.transpose(0, 1)
        v = enc_out.transpose(0, 1)
        attn_out, _ = self.cross_attn(
            q, k, v, key_padding_mask=src_mask
        )
        dec_ctx = dec_h + attn_out.transpose(0, 1)
        logits = self.output_proj(dec_ctx)
        return logits

    def greedy_decode(
        self,
        enc_out,
        src_mask,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ):
        """
        simple greedy decoding (batch=1)
        """
        device = enc_out.device
        B = enc_out.size(0)
        assert B == 1, "greedy_decode only supports batch_size=1 for now."

        tgt_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decode_step(enc_out, src_mask, tgt_ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
            tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break

        return tgt_ids
    
    def beam_decode(
        self,
        enc_out,
        src_mask,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
        beam_size: int = 4,
        len_penalty: float = 0.6,
    ):
        """
        Beam search decoding (batch=1).
        Returns: [1, L] token ids
        """
        device = enc_out.device
        assert enc_out.size(0) == 1, "beam_decode only supports batch_size=1"

        # beams: list of (token_ids_tensor[1,T], logprob_float, ended_bool)
        beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0, False)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, ended in beams:
                if ended:
                    new_beams.append((seq, score, True))
                    continue

                logits = self.decode_step(enc_out, src_mask, seq)         # [1,T,V]
                logp = torch.log_softmax(logits[:, -1, :], dim=-1)        # [1,V]
                topk_logp, topk_ids = torch.topk(logp, k=beam_size, dim=-1)

                for k in range(beam_size):
                    next_id = topk_ids[0, k].view(1, 1)
                    next_seq = torch.cat([seq, next_id], dim=1)
                    next_score = score + float(topk_logp[0, k].item())
                    next_ended = (int(next_id.item()) == eos_id)
                    new_beams.append((next_seq, next_score, next_ended))

            # length penalty ranking
            def norm_score(seq, score):
                L = seq.size(1)
                lp = ((5 + L) / 6) ** len_penalty
                return score / lp

            new_beams.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
            beams = new_beams[:beam_size]

            if all(e for (_, _, e) in beams):
                break

        best_seq, best_score, _ = max(beams, key=lambda x: norm_score(x[0], x[1]))
        return best_seq
