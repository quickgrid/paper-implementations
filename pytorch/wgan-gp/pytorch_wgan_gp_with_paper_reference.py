"""Pytorch WGAN implementation with paper references.

This code builds on top of previous GAN and DCGAN code.
Code structure tries to follow these repositories,
- https://github.com/lucidrains/lightweight-gan/.
- https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py.

Notes:
    - Look into using pathlib for file related tasks.

References:
    - WGAN and WGAN-GP implementation, https://www.youtube.com/watch?v=pG0QZ7OddX4.
    - WGAN paper, https://arxiv.org/abs/1701.07875.
    - WGAN GP paper, https://arxiv.org/abs/1704.00028.
    - DCGAN implementation, https://www.youtube.com/watch?v=IZtv9s_Wx9I.
    - DCGAN paper, https://arxiv.org/abs/1511.06434.
    - GAN implementation, https://www.youtube.com/watch?v=OljTVUVzPpM.
    - GAN paper, https://arxiv.org/abs/1406.2661.
    - WGAN section, https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/.
"""

import os
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


class Critic(nn.Module):
    def __init__(
            self,
            img_channels: int,
            feature_map_base: int,
    ) -> None:
        super(Critic, self).__init__()

        def _blocks(
                in_channels: int,
                out_channels: int,
                kernel_size: Tuple[int, int],
                stride: Tuple[int, int],
                padding: Tuple[int, int],
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2)
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)
        self.C = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_base, k_size, s_amount, p_amount),
            nn.LeakyReLU(0.2),
            *_blocks(feature_map_base, feature_map_base * 2, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 2, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 8, k_size, s_amount, p_amount),
            nn.Conv2d(feature_map_base * 8, 1, k_size, s_amount, padding=(0, 0))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.C(x)


class Generator(nn.Module):
    def __init__(
            self,
            z_dim: int,
            img_channels: int,
            feature_map_base: int,
    ) -> None:
        super(Generator, self).__init__()

        def _blocks(
                in_channels: int,
                out_channels: int,
                kernel_size: Tuple[int, int],
                stride: Tuple[int, int],
                padding: Tuple[int, int],
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)
        self.G = nn.Sequential(
            *_blocks(z_dim, feature_map_base * 16, k_size, stride=(1, 1), padding=(0, 0)),
            *_blocks(feature_map_base * 16, feature_map_base * 8, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 8, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 2, k_size, s_amount, p_amount),
            nn.ConvTranspose2d(feature_map_base * 2, img_channels, k_size, s_amount, p_amount),
            nn.Tanh()
        )

    def forward(self, z):
        return self.G(z)


class CustomImageDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int,
    ) -> None:
        super(CustomImageDataset, self).__init__()
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(image_path)
        return self.transform(image)


class Trainer:
    def __init__(
            self,
            root_dir: str,
            device: torch.device,
            checkpoint_path: str = None,
            save_checkpoint_every: int = 100,
            num_workers: int = 0,
            batch_size: int = 64,
            image_size: int = 64,
            image_channels: int = 3,
            num_epochs: int = 1000,
            z_dim: int = 100,
            learning_rate: float = 1e-4,
            generator_feature_map_base: int = 64,
            critic_feature_map_base: int = 64,
            critic_iterations: int = 2,
            lambda_gp: int = 10,
    ) -> None:
        super(Trainer, self).__init__()
        self.num_epochs = num_epochs
        self.device = device
        self.critic_iterations = critic_iterations
        self.lambda_gp = lambda_gp
        self.BATCH_SIZE = batch_size
        self.z_dim = z_dim
        self.save_checkpoint_every = save_checkpoint_every

        gan_dataset = CustomImageDataset(root_dir=root_dir, image_size=image_size, image_channels=image_channels)
        self.train_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        def _initialize_weights(model, mean=0.0, std=0.02):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                    nn.init.normal_(m.weight.data, mean=mean, std=std)

        self.G = Generator(
            z_dim=z_dim,
            img_channels=image_channels,
            feature_map_base=generator_feature_map_base
        ).to(self.device)

        self.C = Critic(
            img_channels=image_channels,
            feature_map_base=critic_feature_map_base
        ).to(self.device)

        _initialize_weights(self.G)
        _initialize_weights(self.C)

        self.G.train(True)
        self.C.train(True)

        self.opt_G = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        self.opt_C = optim.Adam(self.C.parameters(), lr=learning_rate, betas=(0.0, 0.9))

        # Tensorboard code.
        # Generate tensor directly on device to avoid memory copy.
        # See, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.fixed_noise = torch.randn((32, z_dim, 1, 1), device=device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.step = 0

        self.start_epoch = 0
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def get_gradient_penalty(
            self,
            real: torch.Tensor,
            fake: torch.Tensor
    ) -> torch.Tensor:
        batch_size, c, h, w = real.shape
        epsilon = torch.rand((batch_size, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolated_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = self.C(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.opt_G.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.C.load_state_dict(checkpoint['critic_state_dict'])
        self.opt_C.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            for batch_idx, (real) in enumerate(self.train_loader):
                real = real.to(self.device)
                current_batch_size = real.shape[0]

                # Critic optimization.
                mean_iteration_critic_loss = 0
                for _ in range(self.critic_iterations):
                    noise = torch.randn((current_batch_size, self.z_dim, 1, 1), device=self.device)
                    fake = self.G(noise)
                    critic_real = self.C(real).reshape(-1)
                    critic_fake = self.C(fake.detach()).reshape(-1)
                    gradient_penalty = self.get_gradient_penalty(real=real, fake=fake)
                    loss_critic = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gradient_penalty
                    )
                    mean_iteration_critic_loss += loss_critic.item() / self.critic_iterations
                    self.opt_C.zero_grad()
                    loss_critic.backward()
                    self.opt_C.step()

                # Generator optimization.
                noise = torch.randn((current_batch_size, self.z_dim, 1, 1), device=self.device)
                fake = self.G(noise)
                output = self.C(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                self.opt_G.zero_grad()
                loss_gen.backward()
                self.opt_G.step()

                if batch_idx % self.save_checkpoint_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': self.G.state_dict(),
                        'critic_state_dict': self.C.state_dict(),
                        'generator_optimizer_state_dict': self.opt_G.state_dict(),
                        'critic_optimizer_state_dict': self.opt_C.state_dict(),
                    }, f'checkpoints/checkpoint_{epoch}.pt')

                    self.G.eval()
                    self.C.eval()
                    print(
                        f"EPOCH: [{epoch} / {self.num_epochs}], BATCH: [{batch_idx} / {len(self.train_loader)}], "
                        f"LOSS(Mean) D: {mean_iteration_critic_loss:.4f}, LOSS G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                        self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)
                    self.step += 1
                    self.G.train(True)
                    self.C.train(True)


if __name__ == '__main__':
    trainer = Trainer(
        root_dir=r'C:\data\celeba',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # checkpoint_path='checkpoints/checkpoint_12.pt'
    )
    trainer.train()
