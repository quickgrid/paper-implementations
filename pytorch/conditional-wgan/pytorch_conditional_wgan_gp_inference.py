import matplotlib.pyplot as plt
import torch

from pytorch_conditional_wgan_gp import Generator


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size: int = 64
    image_channels: int = 3
    z_dim: int = 100
    generator_feature_map_base: int = 64
    num_classes: int = 10
    gen_class_embed_size: int = 100
    ncols: int = 6

    noise = torch.randn((ncols, z_dim, 1, 1), device=device)
    labels = torch.LongTensor([2, 9, 7, 1, 5, 3]).to(device)

    gen = Generator(
        z_dim=z_dim,
        img_channels=image_channels,
        feature_map_base=generator_feature_map_base,
        num_classes=num_classes,
        img_size=image_size,
        embed_size=gen_class_embed_size,
    ).to(device)
    gen.eval()

    checkpoint_path = 'checkpoints/checkpoint_146.pt'
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint['generator_state_dict'])

    with torch.no_grad():
        fake = gen(noise, labels)
        fig, ax = plt.subplots(1, ncols)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        for i in range(ncols):
            ax[i].imshow(
                (
                    fake[i].permute(1, 2, 0) * 127.5 + 128
                ).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            )
            ax[i].text(0.0, -2.0, f'Ground Truth: {labels[i]}', fontsize=16)
        plt.show()


if __name__ == '__main__':
    main()
