"""Tests for Imagen code.

TODO:
    - Test cases need to cover expected ouput with optional inputs and various combination.
"""
from typing import Tuple

import torch
from imagen import (
    EfficientUNetResNetBlock,
    EfficientUNetDBlock,
    TransformerEncoderSA,
    EfficientUNetUBlock
)


class Tester:
    def __init__(self, device: str = 'cuda', print_arch: bool = True):
        self.device = device
        self.print_arch = print_arch

    def test_all(self):
        self.test_efficient_unet_res_block()
        self.test_transformer_encoder_sa()
        self.test_efficient_unet_ublock()
        self.test_efficient_unet_dblock()
        self.test_efficient_unet_ublock(stride=(1, 1))
        self.test_efficient_unet_dblock(stride=(1, 1))
        self.test_efficient_unet_ublock(stride=(2, 2))
        self.test_efficient_unet_dblock(stride=(2, 2))
        self.test_efficient_unet_ublock(stride=None)
        self.test_efficient_unet_dblock(stride=None)

    def test_efficient_unet_res_block(
            self,
            n: int = 4,
            in_channels: int = 32,
            out_channels: int = 128,
            hw: int = 32,
            print_arch: bool = None,
    ) -> None:
        print_arch = print_arch if print_arch else self.print_arch

        x = torch.randn(size=(n, in_channels, hw, hw), device=self.device)

        model = EfficientUNetResNetBlock(
            in_channels=in_channels, out_channels=out_channels
        ).to(self.device)

        if print_arch:
            print(model)

        out = model(x)

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (n, out_channels, hw, hw), 'Error expected output shapes do not match.'

    def test_efficient_unet_dblock(
            self,
            n: int = 4,
            out_channels: int = 128,
            hw: int = 32,
            cond_embed_dim: int = 256,
            contextual_text_embed_dim: int = 1024,
            stride: Tuple[int, int] = None,
            print_arch: bool = None,
    ) -> None:
        print_arch = print_arch if print_arch else self.print_arch

        hw_new = hw
        if stride:
            assert stride[0] == stride[1], 'Equal stride must be used.'
            hw_new = hw // stride[0]

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)
        cond_embedding = torch.rand(size=(n, 1, 1, cond_embed_dim), device=self.device)
        contextual_embedding = torch.rand(size=(n, 1, 1, contextual_text_embed_dim), device=self.device)

        model = EfficientUNetDBlock(
            out_channels=out_channels,
            cond_embed_dim=cond_embed_dim,
            contextual_text_embed_dim=contextual_text_embed_dim,
            num_resnet_blocks=2,
            stride=stride,
            use_attention=True,
        ).to(self.device)

        if print_arch:
            print(model)

        out = model(
            x=x, conditional_embedding=cond_embedding, contextual_text_embedding=contextual_embedding
        )

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (n, out_channels, hw_new, hw_new), 'Error expected output shapes do not match.'

    def test_efficient_unet_ublock(
            self,
            n: int = 4,
            out_channels: int = 128,
            hw: int = 32,
            cond_embed_dim: int = 256,
            stride: Tuple[int, int] = None,
            print_arch: bool = None,
    ) -> None:
        print_arch = print_arch if print_arch else self.print_arch

        hw_new = hw
        if stride:
            assert stride[0] == stride[1], 'Equal stride must be used.'
            hw_new = hw * stride[0]

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)
        cond_embedding = torch.rand(size=(n, 1, 1, cond_embed_dim), device=self.device)

        model = EfficientUNetUBlock(
            out_channels=out_channels,
            cond_embed_dim=cond_embed_dim,
            num_resnet_blocks=3,
            stride=stride,
            use_attention=True,
        ).to(self.device)

        if print_arch:
            print(model)

        out = model(
            x=x, x_skip=x, conditional_embedding=cond_embedding
        )

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (n, out_channels, hw_new, hw_new), 'Error expected output shapes do not match.'

    def test_transformer_encoder_sa(
            self,
            n: int = 4,
            out_channels: int = 128,
            hw: int = 32,
            print_arch: bool = None,
    ) -> None:
        print_arch = print_arch if print_arch else self.print_arch

        x = torch.randn(size=(n, out_channels, hw, hw), device=self.device)

        model = TransformerEncoderSA(
            num_channels=out_channels,
        ).to(self.device)

        if print_arch:
            print(model)

        out = model(x)

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (n, out_channels, hw, hw), 'Error expected output shapes do not match.'

        
if __name__ == '__main__':
    tester = Tester(print_arch=False)
    tester.test_all()
    # tester.test_efficient_unet_res_block()
    # tester.test_efficient_unet_dblock()
    # tester.test_transformer_encoder_sa()
    # tester.test_efficient_unet_ublock()
