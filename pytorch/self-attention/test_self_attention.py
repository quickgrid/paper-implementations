import torch
from self_attention import TransformerEncoderCustomSA, TransformerEncoderSA


class Tester:
    def __init__(self, device: str = 'cuda', print_arch: bool = False):
        self.print_arch = print_arch
        self.device = device

    def test_transformer_encoder_sa(
            self,
            minibatch_size: int = 4,
            in_channels: int = 128,
            hw: int = 64,
            print_arch: bool = None,
    ) -> None:
        x = torch.randn(size=(minibatch_size, in_channels, hw, hw), device=self.device)
        model = TransformerEncoderSA(num_channels=in_channels).to(self.device)

        print(f'Param count: {sum([p.numel() for p in model.parameters()])}')

        if print_arch:
            print(model)

        out = model(x)

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (minibatch_size, in_channels, hw, hw), 'Error expected output shapes do not match.'

    def test_transformer_encoder_custom_sa(
            self,
            minibatch_size: int = 4,
            in_channels: int = 128,
            hw: int = 64,
            print_arch: bool = None,
    ) -> None:
        x = torch.randn(size=(minibatch_size, in_channels, hw, hw), device=self.device)
        model = TransformerEncoderCustomSA(num_channels=in_channels).to(self.device)

        print(f'Param count: {sum([p.numel() for p in model.parameters()])}')

        if print_arch:
            print(model)

        out = model(x)

        print(f'In shape: {x.shape}, Out shape: {out.shape}')
        assert out.shape == (minibatch_size, in_channels, hw, hw), 'Error expected output shapes do not match.'


if __name__ == '__main__':
    tester = Tester()
    tester.test_transformer_encoder_sa(print_arch=True)
    tester.test_transformer_encoder_custom_sa(print_arch=True)
