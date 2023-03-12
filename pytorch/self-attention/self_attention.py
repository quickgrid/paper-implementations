"""Attention is all you need paper implementation of Transformers and Multi-Head Self Attention.

Implemention is not tested in training and not sure if it will work and is correct. It can be
imported into other modules.

References:
    - Transformers, Self Attention paper, https://arxiv.org/abs/1706.03762.
    - http://nlp.seas.harvard.edu/annotated-transformer/
"""
import math

import torch
from torch import nn
from torch.functional import F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = dropout

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
    ) -> [torch.Tensor, torch.Tensor]:
        """Implementation of figure 2 and equation 1. In section 3.2.3 masking is acheived by setting it to -inf.
        Here, -1e9 is used.

        Args:
            query: Shape of (batch, num_tokens, query_dim).
            key:
            value:
            mask:
        """
        d_k = key.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(d_k)

        if mask is not None:
            scaled_scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scaled_scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout)

        output = torch.matmul(attention, value)
        return output, attention


class MultiHeadSelfAttentionFused(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_dim: int = None,
            num_heads: int = 8,
            head_dim: int = 64,
            bias: bool = True,
            dropout: float = 0.
    ):
        """Multi Head Self Attention from attention is all you need paper, https://arxiv.org/abs/1706.03762.
        Figure 2 shows the MSA or MHSA architecture. Mask not implemented.

        Section 3.2.2 mentions default number of heads or parallel attention layer `h = 8`. For each of these head
        paper output dimension is 64. It mentions query, key, values should be linearly projected `h` times with
        learned projection dimension `d_q`, `d_k`, `d_v`.

        Scaling is defined in section 3.2.1 as `1/sqrt(d_k)`.

        Args:
            embed_dim: Output embedding dimension or query dimension.
            context_dim: Dimension of text embedding to use for multimodal case.
            num_heads: Parallel attention layer or head defined.
            head_dim: Dimension per head.
            dropout: Dropout applied to attention.
        """
        super(MultiHeadSelfAttentionFused, self).__init__()
        self.dropout = dropout
        hidden_dim = num_heads * head_dim
        context_dim = context_dim if context_dim is not None else embed_dim
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(in_features=embed_dim, out_features=hidden_dim, bias=bias)
        self.k_linear = nn.Linear(in_features=context_dim, out_features=hidden_dim, bias=bias)
        self.v_linear = nn.Linear(in_features=context_dim, out_features=hidden_dim, bias=bias)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Mutlihead self attention for regular implementation and context implementation. When context is provided
        as for imagen both key and value should be sent same contextual text embedding.

        Attention is calculculated based on equation 1 of attention paper.
        """
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        attention_qkv = torch.matmul(q, k.permute(0, 2, 1)) * self.scale
        attention_qkv = F.dropout(F.softmax(attention_qkv, dim=1), p=self.dropout)
        attention_qkv = torch.matmul(attention_qkv, v)
        return self.output_linear(attention_qkv)


class TransformerEncoderCustomSA(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_heads: int = 8,
            hidden_dim: int = None,
            dropout: int = 0.0,
    ):
        """A custom implementation of transformer encoder with usage of custom multihead self attention. Layer norm
        is applied before MHA and feed forward as prenorm.

        Second layer norm is merged feed forward.
        """
        super(TransformerEncoderCustomSA, self).__init__()

        per_head_dim = num_channels // num_heads
        self.mha = MultiHeadSelfAttentionFused(
            embed_dim=num_channels, num_heads=num_heads, head_dim=per_head_dim, bias=True,
        )

        self.ln = nn.LayerNorm([num_channels])

        hidden_dim = hidden_dim if hidden_dim is not None else (num_channels * 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=num_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attention_value + x
        x = self.mlp(x) + x
        return x.permute(0, 2, 1).view(b, c, h, w)


class TransformerEncoderSA(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_heads: int = 8,
            hidden_dim: int = None,
            dropout: int = 0.0,
    ):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
        https://arxiv.org/abs/1706.03762.
        """
        super(TransformerEncoderSA, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])

        hidden_dim = hidden_dim if hidden_dim is not None else (num_channels * 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=num_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attention_value + x
        x = self.mlp(x) + x
        return x.permute(0, 2, 1).view(b, c, h, w)
