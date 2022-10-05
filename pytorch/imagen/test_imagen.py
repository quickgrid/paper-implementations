"""Tests for Imagen code.

TODO:
    - Test cases need to cover expected ouput with optional inputs and various combination.
"""

import torch
from imagen import (
    EfficientUNetResNetBlock,
    EfficientUNetDBlock,
    TransformerEncoderSA,
    EfficientUNetUBlock
)


class Tester:
    def __init__(self, device: str = 'cuda'):
        self.device = device

    def test_all(self) -> None:
        self.test_efficient_unet_res_block()
        self.test_efficient_unet_dblock()
        self.test_transformer_encoder_sa()
        self.test_efficient_unet_ublock()

    def test_efficient_unet_res_block(self) -> None:
        n = 4
        in_channels = 32
        out_channels = 128
        hw = 64

        x = torch.randn(size=(n, in_channels, hw, hw), device=self.device)

        efficient_unet_res_block = EfficientUNetResNetBlock(
            in_channels=in_channels, out_channels=out_channels
        ).to(self.device)

        print(efficient_unet_res_block)

        out = efficient_unet_res_block(x)
        print(f'In shape: {x.shape}, Out shape: {out.shape}')

        assert out.shape == (n, out_channels, hw, hw), 'Error expected output shapes do not match.'

    def test_efficient_unet_dblock(self) -> None:
        n = 4
        out_channels = 128
        hw = 64
        cond_embed_dim = 256
        contextual_text_embed_dim = 1024

        stride = (2, 2)
        assert stride[0] == stride[1], 'Equal stride must be used.'
        hw_new = 64 // stride[0]

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)
        cond_embedding = torch.rand(size=(n, 1, 1, cond_embed_dim), device=self.device)
        contextual_embedding = torch.rand(size=(n, 1, 1, contextual_text_embed_dim), device=self.device)

        efficient_unet_dblock = EfficientUNetDBlock(
            out_channels=out_channels,
            cond_embed_dim=cond_embed_dim,
            contextual_text_embed_dim=contextual_text_embed_dim,
            num_resnet_blocks=2,
            stride=stride,
            use_attention=True,
        ).to(self.device)

        print(efficient_unet_dblock)

        out = efficient_unet_dblock(
            x=x, conditional_embedding=cond_embedding, contextual_text_embedding=contextual_embedding
        )
        print(f'In shape: {x.shape}, Out shape: {out.shape}')

        assert out.shape == (n, out_channels, hw_new, hw_new), 'Error expected output shapes do not match.'

    def test_efficient_unet_ublock(self):
        n = 4
        out_channels = 128
        hw = 64
        cond_embed_dim = 256

        stride = (2, 2)
        assert stride[0] == stride[1], 'Equal stride must be used.'
        hw_new = 64 * stride[0]

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)
        cond_embedding = torch.rand(size=(n, 1, 1, cond_embed_dim), device=self.device)

        efficient_unet_ublock = EfficientUNetUBlock(
            out_channels=out_channels,
            cond_embed_dim=cond_embed_dim,
            num_resnet_blocks=3,
            stride=stride,
            use_attention=True,
        ).to(self.device)

        print(efficient_unet_ublock)

        out = efficient_unet_ublock(
            x=x, x_skip=x, conditional_embedding=cond_embedding
        )
        print(f'In shape: {x.shape}, Out shape: {out.shape}')

        assert out.shape == (n, out_channels, hw_new, hw_new), 'Error expected output shapes do not match.'

    def test_transformer_encoder_sa(self) -> None:
        n = 4
        out_channels = 128
        hw = 64

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)

        transformer_encoder_sa = TransformerEncoderSA(
            num_channels=out_channels,
        ).to(self.device)

        print(transformer_encoder_sa)

        out = transformer_encoder_sa(x)
        print(f'In shape: {x.shape}, Out shape: {out.shape}')

        assert out.shape == (n, out_channels, hw, hw), 'Error expected output shapes do not match.'


if __name__ == '__main__':
    tester = Tester()
    tester.test_all()
    # tester.test_efficient_unet_res_block()
    # tester.test_efficient_unet_dblock()
    # tester.test_transformer_encoder_sa()
    # tester.test_efficient_unet_ublock()
