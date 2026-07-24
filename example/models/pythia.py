import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Configuration for Pythia-160m
# ---------------------------
class PythiaConfig:
    vocab_size = 50257      # GPT-style vocabulary
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    seq_len = 256
    dropout = 0.1
    layer_norm_eps = 1e-5

# ---------------------------
# Causal Self-Attention
# ---------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.seq_len, config.seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.out(out))
        return out

# ---------------------------
# Transformer Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ---------------------------
# Pythia Model
# ---------------------------
class PythiaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # [B, T, vocab_size]
