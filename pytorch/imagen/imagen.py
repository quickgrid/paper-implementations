"""Imagen implementation.

TODO: Add gradient checkpoint for chosen modules.

References:
    - Imagen paper, https://arxiv.org/abs/2205.11487.
"""
from typing import Tuple, Union

import numpy as np
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

        hidden_dim = hidden_dim if hidden_dim is not None else (num_channels * 2)
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
            cond_embed_dim: int,
            num_resnet_blocks: int,
            contextual_text_embed_dim: int = None,
            stride: Tuple[int, int] = None,
            use_attention: bool = False,
    ):
        """Implementation of Efficient UNet DBlock as shown in Figure A.28. If stide is provided downsamples
        input tensor by the amount.

        Embedding layers are used to bring feature map shape to expected embedding dimension from different input
        dimensions.

        In paper first conv in DBlock is optional with strided downsampling. Conv block is kept and only down samples
        when stride is provided else keeps same shape in h, w.

        Args:
            out_channels: Current block expected output channels.
            num_resnet_blocks: Number of sequential resnet blocks in dblock between CombineEmbs and SelfAttention.
            cond_embed_dim: Conditinal embeddings dimension like time, class, text embeddings.
            contextual_text_embed_dim: Embedded text dimension for example, T5 output with 1024 in channel dimension.
            stride: With (1, 1) output has same h, w as input with shape of (batch_size, out_channel, h, w).
                With stride of (2, 2) downsamples tensor as (batch_size, out_channel, h / 2, w / 2).
            use_attention: Attention is only used if True.
        """
        super(EfficientUNetDBlock, self).__init__()
        self.use_attention = use_attention
        self.use_conv = True if stride is not None else False

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )

        self.conditional_embedding_layer = nn.Sequential(
            nn.Linear(in_features=cond_embed_dim, out_features=out_channels)
        )

        self.resnet_blocks = nn.Sequential()
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                EfficientUNetResNetBlock(in_channels=out_channels, out_channels=out_channels)
            )

        if use_attention:
            self.transformer_encoder_sa = TransformerEncoderSA(num_channels=out_channels)
            self.contextual_text_embedding_layer = nn.Sequential(
                nn.Linear(in_features=contextual_text_embed_dim, out_features=out_channels)
            )

    def forward(
            self,
            x: torch.Tensor,
            conditional_embedding: torch.Tensor,
            contextual_text_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """DBlock, initial conv (optional) -> combine embs -> resnet blocks -> self attention (optional).

        Expected conditional_embedding shape (batch, 1, 1, cond_embed_dim), which is passed through embedding layer.
        Embedding layer converts cond_embed_dim to out_channels to match initial conv output shape. The output shape
        is (batch, 1, 1, out_channels).

        The conditional embedding, `cond_embed` is reshaped to add with input feature map x. Channel first
        format used in the code, but channel last can be used by reshaping x. Out channel feature map is replicated
        along height and width per pixel. If shape of output of initial conv is (batch, out_channels, hw, hw) then
        `cond_embed` is converted to,
        (batch, 1, 1, out_channels) -> (batch, out_channels, 1, 1) -> (batch, out_channels, hw, hw).

        Attention is only used if defined for that layer.

        Input `conditional_embedding` and `contextual_text_embedding` shape in channel dimension do not need to be
        same as they are projected to expected shape `output_channels` with embedding layers.

        Args:
            x: Previous DBlock output.
            conditional_embedding: Time, class, pooled text embedding. Example shape, (batch, 1, 1, 256).
            contextual_text_embedding: Contextual text embedding from pretrained model like T5. Example shape,
                (batch, 1, 1, 1024).
        """
        x = self.initial_conv(x) if self.use_conv else x
        cond_embed = self.conditional_embedding_layer(conditional_embedding)
        cond_embed = cond_embed.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + cond_embed
        x = self.resnet_blocks(x)

        if self.use_attention:
            context_text_embed = self.contextual_text_embedding_layer(contextual_text_embedding)
            context_text_embed = context_text_embed.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + context_text_embed
            x = self.transformer_encoder_sa(x)

        return x


class EfficientUNetUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_embed_dim: int,
            num_resnet_blocks: int,
            stride: Tuple[int, int] = None,
            use_attention: bool = False,
    ):
        """Implementation of Efficient UNet UBlock as shown in Figure A.29.

        Rather than not having conv block when stride is not provided it is kept. It upsamples if stride is provided
        else keeps the same shape in spatial dimension.
        """
        super(EfficientUNetUBlock, self).__init__()
        self.use_attention = use_attention
        self.use_conv = True if stride is not None else False

        self.conditional_embedding_layer = nn.Sequential(
            nn.Linear(in_features=cond_embed_dim, out_features=out_channels)
        )

        self.input_embedding_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
        )

        self.resnet_blocks = nn.Sequential()
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                EfficientUNetResNetBlock(in_channels=out_channels, out_channels=out_channels)
            )

        if use_attention:
            self.transformer_encoder_sa = TransformerEncoderSA(num_channels=out_channels)

        self.last_conv_upsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
            ),
            nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
        )

    def forward(
            self,
            x: torch.Tensor,
            conditional_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Ublock, combine embs -> resnet blocks -> self attention (optional) -> last conv (optional).

        Args:
            x: Previous UBlock output.
            conditional_embedding: Time, class, pooled Text embeddings.
        """
        cond_embed = self.conditional_embedding_layer(conditional_embedding)
        cond_embed = cond_embed.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.input_embedding_layer(x)

        x = x + cond_embed
        x = self.resnet_blocks(x)
        x = self.transformer_encoder_sa(x) if self.use_attention else x
        x = self.last_conv_upsampler(x) if self.use_conv else x
        return x


class EfficientUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            cond_embed_dim: int = 512,
            base_channel_dim: int = 32,
            use_attention: bool = True,
            num_resnet_blocks: Union[Tuple[int, ...], int] = None,
            channel_mults: Tuple[int, ...] = None,
    ):
        """UNet implementation for 64 x 64 image as defined in Section F.1 and efficient UNet architecture for
         64 -> 256 upsampling as shown in Figure A.30.

        Ellipsis used for variable number of U and D blocks. The number of D and U blocks depend on the number
        of `channel_mults`.

        Parameter of the current UNet model does not depend on the image resolution.

        TODO: In cascade diffusion may need to concat low res image with noisy image which should result in 6 channels.

        Args:
            in_channels: Input image tensor channels.
            cond_embed_dim: Timestep or text embedding output dimension.
            base_channel_dim: Base value for multiplying with channel_mults for U or D blocks of UNet.
            num_resnet_blocks: Number of resnet blocks in each of the U or D blocks of UNet.
            channel_mults: Multiplier values for each of the U or D blocks in UNet.
        """
        super(EfficientUNet, self).__init__()
        if channel_mults is None:
            channel_mults = (1, 2, 3, 4)
        if num_resnet_blocks is None:
            num_resnet_blocks = 3

        if isinstance(num_resnet_blocks, int):
            num_resnet_blocks = (num_resnet_blocks, ) * len(channel_mults)

        assert len(channel_mults) == len(num_resnet_blocks), 'channel_mults and num_resnet_blocks should be same shape.'

        mutliplied_channels = np.array(channel_mults) * base_channel_dim
        mutliplied_channels_len = len(mutliplied_channels)
        mutliplied_channels_reversed = np.flip(mutliplied_channels)
        num_resnet_blocks_reversed = np.flip(num_resnet_blocks)

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mutliplied_channels[0],
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.dblocks = nn.ModuleList()
        for idx, num_channels in enumerate(mutliplied_channels[:-1]):
            self.dblocks.append(
                EfficientUNetDBlock(
                    in_channels=mutliplied_channels[idx],
                    out_channels=mutliplied_channels[idx + 1],
                    cond_embed_dim=cond_embed_dim,
                    num_resnet_blocks=num_resnet_blocks[idx],
                    stride=(2, 2),
                )
            )

        self.ublocks = nn.ModuleList()
        self.ublocks.append(
            EfficientUNetUBlock(
                in_channels=mutliplied_channels_reversed[0],
                out_channels=mutliplied_channels_reversed[1],
                cond_embed_dim=cond_embed_dim,
                num_resnet_blocks=num_resnet_blocks_reversed[1],
                stride=(2, 2),
                use_attention=use_attention,
            )
        )
        for idx in range(1, mutliplied_channels_len - 1, 1):
            self.ublocks.append(
                EfficientUNetUBlock(
                    in_channels=mutliplied_channels_reversed[idx] * 2,
                    out_channels=mutliplied_channels_reversed[idx + 1],
                    cond_embed_dim=cond_embed_dim,
                    num_resnet_blocks=num_resnet_blocks_reversed[idx],
                    stride=(2, 2),
                    use_attention=use_attention,
                )
            )

        self.image_projection = nn.Conv2d(
            in_channels=channel_mults[0] * base_channel_dim, out_channels=3, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(
            self,
            x: torch.Tensor,
            conditional_embedding: torch.Tensor,
            contextual_text_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Efficient UNet forward for given number of unet blocks.

        As shown in Figure A.30 the last unet dblock and first unet block in the middle do not have skip connection.
        """
        x = self.initial_conv(x)

        x_skip_outputs = []
        for dblock in self.dblocks:
            x = dblock(x, conditional_embedding, contextual_text_embedding)
            x_skip_outputs.append(x)

        x_skip_outputs.pop()
        x = self.ublocks[0](x=x, conditional_embedding=conditional_embedding)

        for ublock in self.ublocks[1:]:
            x = torch.cat((x, x_skip_outputs.pop()), dim=1)
            x = ublock(x=x, conditional_embedding=conditional_embedding)

        x = self.image_projection(x)
        return x
