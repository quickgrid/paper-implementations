"""Imagen implementation.

References:
    - Imagen paper, https://arxiv.org/abs/2205.11487.
"""
from typing import Tuple

import torch
from torch import nn


class EfficientUNetResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int = 8,
    ):
        """Efficient UNet implementation from Figure A.27.

        Input channels are split in `num_groups` with each group having `in_channels / num_groups` channels. Groupnorm,
        https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html.

        SiLU is used in place of Swish as both are same functions,
        https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html.
        """
        super(EfficientUNetResNetBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.skip_path = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_path(x) + self.skip_path(x)


class TransformerEncoderSA(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_heads: int = 8,
            hidden_dim: int = None,
            dropout: int = 0.0,
    ):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
         https://arxiv.org/pdf/2010.11929.pdf.
        """
        super(TransformerEncoderSA, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm([num_channels])
        self.ln_2 = nn.LayerNorm([num_channels])

        hidden_dim = hidden_dim if hidden_dim else (num_channels * 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=num_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x_ln = self.ln_1(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attention_value + x
        x = self.mlp(self.ln_2(x)) + x
        return x.permute(0, 2, 1).view(b, c, h, w)


class EfficientUNetDBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_resnet_blocks: int,
            embed_dim: int,
            stride: Tuple[int, int],
            use_attention: bool = False,
    ):
        """Implementation of Efficient UNet DBlock as shown in Figure A.28.

        Args:
            in_channels: Previous layer output channels.
            out_channels: Current block expected output channels.
            num_resnet_blocks: Number of sequential resnet blocks in dblock between CombineEmbs and SelfAttention.
            embed_dim: Conditinal embeddings dimension like time, class, text embeddings.
            stride: With (1, 1) output has same h, w as input with shape of (batch_size, out_channel, h, w).
                With stride of (2, 2) downsamples tensor as (batch_size, out_channel, h / 2, w / 2).
        """
        super(EfficientUNetDBlock, self).__init__()
        self.use_attention = use_attention

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )
        self.conditional_embedding_layer = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=out_channels)
        )

        self.resnet_blocks = nn.Sequential()
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                EfficientUNetResNetBlock(in_channels=out_channels, out_channels=out_channels)
            )

        if use_attention:
            self.transformer_encoder_sa = TransformerEncoderSA(num_channels=out_channels)

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        """DBlock, initial conv -> combine embs -> resnet blocks -> self attention.

        Expected t_embedding shape (batch, 1, 1, embed_dim), which is passed through embedding layer.
        Embedding layer converts embed_dim to out_channels to match initial conv output shape. The output shape
        is (batch, 1, 1, out_channels).

        The time embedding, t is reshaped to add with input feature map x. Channel first format used in the code, but
        channel last can be used by reshaping x. Out channel feature map is replicated along height and width per pixel.
        If shape of output of initial conv is (batch, out_channels, hw, hw) then t is converted to,
        (batch, 1, 1, out_channels) -> (batch, out_channels, 1, 1) -> (batch, out_channels, hw, hw).

        Attention is only used if defined for that layer.
        """
        x = self.initial_conv(x)
        t = self.conditional_embedding_layer(t_embedding)
        t = t.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + t
        x = self.resnet_blocks(x)
        return self.transformer_encoder_sa(x) if self.use_attention else x
