"""Ember: a decoder-only transformer.

Pre-norm RMSNorm, RoPE positions, SwiGLU FFN, no biases anywhere,
weight-tied embedding/unembedding, causal masking.

Attention has two switchable implementations:
  - "naive": explicit matmul -> causal mask -> softmax (the reference)
  - "sdpa":  torch.nn.functional.scaled_dot_product_attention (what trains)
Tests assert they are equivalent.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmberConfig:
    vocab_size: int = 8192
    ctx_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_hidden: int = 1408
    rope_theta: float = 10000.0
    attn_impl: str = "sdpa"  # "sdpa" | "naive"
    # Ablation axes (defaults = the Ember stack; plan section 3B)
    pos_encoding: str = "rope"  # "rope" | "learned"
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    norm_placement: str = "pre"  # "pre" | "post"
    ffn_type: str = "swiglu"  # "swiglu" | "gelu"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Closed-form parameter count (embedding tied with unembedding)."""
        d, f = self.d_model, self.ffn_hidden
        ffn = 3 * d * f if self.ffn_type == "swiglu" else 2 * d * f
        per_layer = 4 * d * d + ffn + 2 * d  # qkvo + ffn + two norms
        total = self.vocab_size * d + self.n_layers * per_layer + d  # + final norm
        if self.pos_encoding == "learned":
            total += self.ctx_len * d
        return total


# The four sizes of the scaling study (plan section 2.2), plus the
# depth-vs-width twin: same ~13.8M params as ember-12m, 12x272 vs 6x384.
PRESETS = {
    "ember-5m": dict(d_model=256, n_layers=4, n_heads=4, ffn_hidden=704),
    "ember-12m": dict(d_model=384, n_layers=6, n_heads=6, ffn_hidden=1024),
    "ember-12m-deep": dict(d_model=272, n_layers=12, n_heads=4, ffn_hidden=815),
    "ember-30m": dict(d_model=512, n_layers=8, n_heads=8, ffn_hidden=1408),
    "ember-60m": dict(d_model=640, n_layers=10, n_heads=10, ffn_hidden=1728),
}


def preset_config(name: str, **overrides) -> EmberConfig:
    return EmberConfig(**{**PRESETS[name.lower()], **overrides})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability under bf16 autocast.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class RoPE(nn.Module):
    """Rotary position embeddings, rotate-half (GPT-NeoX) convention."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0):
        # q, k: (B, H, T, head_dim)
        T = q.shape[-2]
        cos = self.cos[pos_offset : pos_offset + T]
        sin = self.sin[pos_offset : pos_offset + T]
        qf, kf = q.float(), k.float()
        q_rot = qf * cos + self._rotate_half(qf) * sin
        k_rot = kf * cos + self._rotate_half(kf) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


class KVCache:
    """Hand-rolled per-layer K/V cache for incremental decoding.

    Buffers are allocated lazily on first append so dtype/device follow the
    K tensors the model actually produces (fp32, or bf16 under autocast).
    Memory: 2 (K,V) * layers * heads * head_dim * ctx_len * bytes/elem
    (~8 MB per sequence for ember-30M at ctx 512 in bf16).
    """

    def __init__(self, cfg: EmberConfig, batch_size: int):
        self.cfg = cfg
        self.batch_size = batch_size
        self.buf: torch.Tensor | None = None
        self.pos = 0  # tokens already cached

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Write this layer's new K/V at the current position; return full views."""
        if self.buf is None:
            c = self.cfg
            self.buf = torch.zeros(
                c.n_layers, 2, self.batch_size, c.n_heads, c.ctx_len, c.head_dim,
                dtype=k.dtype, device=k.device,
            )
        T = k.shape[-2]
        assert self.pos + T <= self.cfg.ctx_len, "KV cache overflow past ctx_len"
        self.buf[layer, 0, :, :, self.pos : self.pos + T] = k
        self.buf[layer, 1, :, :, self.pos : self.pos + T] = v
        end = self.pos + T
        return self.buf[layer, 0, :, :, :end], self.buf[layer, 1, :, :, :end]

    def advance(self, T: int):
        """Called once per forward, after every layer has appended."""
        self.pos += T

    @property
    def nbytes(self) -> int:
        return 0 if self.buf is None else self.buf.numel() * self.buf.element_size()


def _offset_causal_mask(T: int, S: int, device) -> torch.Tensor:
    """Bool mask (T, S), True = attend. Query i sits at position S - T + i."""
    q_pos = torch.arange(S - T, S, device=device).unsqueeze(1)
    k_pos = torch.arange(S, device=device).unsqueeze(0)
    return k_pos <= q_pos


class Attention(nn.Module):
    def __init__(self, cfg: EmberConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d = cfg.d_model
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rope = (
            RoPE(cfg.head_dim, cfg.ctx_len, cfg.rope_theta)
            if cfg.pos_encoding == "rope"
            else None
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, T, d = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)
        if self.rope is not None:
            pos = cache.pos if cache is not None else 0
            q, k = self.rope(q, k, pos_offset=pos)

        if cache is not None:
            k, v = cache.append(self.layer_idx, k, v)
        S = k.shape[-2]  # total keys visible (cached + new)

        if self.cfg.attn_impl == "sdpa":
            if T == S:  # plain training/prefill forward
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif T == 1:  # incremental decode: last query sees everything
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:  # chunked append onto a non-empty cache
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=_offset_causal_mask(T, S, x.device)
                )
        elif self.cfg.attn_impl == "naive":
            scores = q @ k.transpose(-2, -1) / math.sqrt(hd)  # (B, H, T, S)
            scores = scores.masked_fill(~_offset_causal_mask(T, S, x.device), float("-inf"))
            out = scores.softmax(dim=-1) @ v
        else:
            raise ValueError(f"unknown attn_impl: {self.cfg.attn_impl}")

        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: EmberConfig):
        super().__init__()
        d, f = cfg.d_model, cfg.ffn_hidden
        self.w1 = nn.Linear(d, f, bias=False)  # gate
        self.w3 = nn.Linear(d, f, bias=False)  # up
        self.w2 = nn.Linear(f, d, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    """Ablation FFN. Match params to SwiGLU by setting ffn_hidden 1.5x."""

    def __init__(self, cfg: EmberConfig):
        super().__init__()
        d, f = cfg.d_model, cfg.ffn_hidden
        self.w1 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def make_norm(cfg: EmberConfig) -> nn.Module:
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.d_model)
    return nn.LayerNorm(cfg.d_model, bias=False)


class Block(nn.Module):
    def __init__(self, cfg: EmberConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.ffn_type == "swiglu" else GeluMLP(cfg)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        if self.cfg.norm_placement == "pre":
            x = x + self.attn(self.norm1(x), cache=cache)
            x = x + self.mlp(self.norm2(x))
        else:  # post-norm (original transformer; expected to train worse)
            x = self.norm1(x + self.attn(x, cache=cache))
            x = self.norm2(x + self.mlp(x))
        return x


class Ember(nn.Module):
    def __init__(self, cfg: EmberConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.ctx_len, cfg.d_model) if cfg.pos_encoding == "learned" else None
        )
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Scaled init on residual-stream projections (GPT-2 style).
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=resid_std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ):
        B, T = idx.shape
        pos = cache.pos if cache is not None else 0
        assert pos + T <= self.cfg.ctx_len, (
            f"overflow: position {pos}+{T} > ctx_len {self.cfg.ctx_len}"
        )
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(pos, pos + T, device=idx.device))
        for block in self.blocks:
            x = block(x, cache=cache)
        if cache is not None:
            cache.advance(T)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size).float(), targets.view(-1)
            )
        return logits, loss

    def _sample_next(self, logits, temperature, top_k, top_p):
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Keep the smallest set whose cumulative prob exceeds top_p
            # (shift right so the token that crosses the threshold survives).
            drop = (cum > top_p).roll(1, dims=-1)
            drop[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits
            )
        return torch.multinomial(logits.softmax(dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature=0 means greedy.

        With use_cache, the prompt is prefilled once and each new token costs
        a single-position forward. Falls back to the recompute-everything path
        when the request wouldn't fit in ctx_len (the cache can't slide).
        """
        was_training = self.training
        self.eval()
        try:
            if use_cache and idx.shape[1] + max_new_tokens <= self.cfg.ctx_len:
                cache = KVCache(self.cfg, idx.shape[0])
                logits, _ = self(idx, cache=cache)  # prefill
                for i in range(max_new_tokens):
                    if i > 0:
                        logits, _ = self(next_id, cache=cache)  # decode one position
                    next_id = self._sample_next(
                        logits[:, -1, :], temperature, top_k, top_p
                    )
                    idx = torch.cat([idx, next_id], dim=1)
                    if eos_id is not None and (next_id == eos_id).all():
                        break
                return idx

            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.ctx_len :]
                logits, _ = self(idx_cond)
                next_id = self._sample_next(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, next_id], dim=1)
                if eos_id is not None and (next_id == eos_id).all():
                    break
            return idx
        finally:
            self.train(was_training)
