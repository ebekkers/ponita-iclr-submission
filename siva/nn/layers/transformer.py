import torch
import math
import torch.nn.functional as F
import torch.nn as nn


def scaled_dot_product(
    q, k, v, mask=None, key_padding_mask=None, attn_bias=None, clamp=5
):
    EPS = -9e15
    assert mask is None or key_padding_mask is None, "Please don't use multiple masks"

    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))

    if attn_bias is not None:
        attn_logits += attn_bias
    attn_logits = attn_logits / math.sqrt(d_k)

    # Clamping seems to work good
    if clamp is not None:
        attn_logits = torch.clamp(attn_logits, -clamp, clamp)

    # Apply full mask or key_padding_mask [B, L, L] or [B, L]
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask[:,None], EPS)
    if key_padding_mask is not None:
        attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, None, :], EPS)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)

    # return values, attention
    return values, None


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, key_padding_mask=None, attn_bias=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(
            q, k, v, mask=mask, key_padding_mask=key_padding_mask, attn_bias=attn_bias
        )
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        return o


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, d_bias=0, dropout=0.0, act=nn.ReLU):
        super().__init__()

        self.mha_norm = nn.LayerNorm(d_model)
        self.mha = MultiheadAttention(d_model, n_heads)
        # self.mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            act(),
            nn.Linear(d_ffn, d_model),
        )

        if d_bias > 0:
            self.bias_net = nn.Sequential(
                nn.Linear(d_ffn, n_heads),
            )

        if dropout > 0:
            self.mha_dropout = nn.Dropout(dropout)
            self.ffn_dropout = nn.Dropout(dropout)
        else:
            self.mha_dropout = None
            self.ffn_dropout = None

    def forward(self, x, mask=None, key_padding_mask=None, attn_bias=None):
        if attn_bias is not None:
            attn_bias = self.bias_net(attn_bias)
            attn_bias = attn_bias.permute(0, 3, 1, 2)

        z = self.mha_norm(x)
        z = self.mha(
            z, mask=mask, key_padding_mask=key_padding_mask, attn_bias=attn_bias
        )
        # z, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask.logical_not())
        if self.mha_dropout:
            z = self.mha_dropout(z)
        x = x + z

        z = self.ffn_norm(x)
        z = self.ffn(z)
        if self.ffn_dropout:
            z = self.ffn_dropout(z)
        x = x + z
        return x