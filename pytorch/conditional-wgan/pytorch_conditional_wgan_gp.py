"""Pytorch class conditional WGAN GP implementation.

Notes
    - Test on other datasets.
    - Understand more clearly and simplify code.

References
    - WGAN and WGAN-GP implementation, https://www.youtube.com/watch?v=pG0QZ7OddX4.
    - WGAN paper, https://arxiv.org/abs/1701.07875.
    - WGAN GP paper, https://arxiv.org/abs/1704.00028.
    - Conditional GAN implementation, https://www.youtube.com/watch?v=Hp-jWm2SzR8.
    - https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP
"""

import os

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
            num_classes: int,
            img_size: int,
    ) -> None:
        super(Critic, self).__init__()

        def _blocks(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2),
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)
        self.C = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_map_base, k_size, s_amount, p_amount),
            nn.LeakyReLU(0.2),
            *_blocks(feature_map_base, feature_map_base * 2, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 2, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 8, k_size, s_amount, p_amount),
            nn.Conv2d(feature_map_base * 8, 1, k_size, s_amount, padding=(0, 0))
        )

        self.img_size = img_size
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=img_size * img_size)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Reshape embedding to single channel per class with (img_size * img_size) to match shape and concat.
        """
        class_embedding = self.class_embedding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, class_embedding], dim=1)  # Merge in channel dimension of N C H W.
        return self.C(x)


class Generator(nn.Module):
    def __init__(
            self,
            z_dim: int,
            img_channels: int,
            feature_map_base: int,
            num_classes: int,
            img_size: int,
            embed_size: int,
    ) -> None:
        super(Generator, self).__init__()

        def _blocks(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)
        self.G = nn.Sequential(
            *_blocks(z_dim + embed_size, feature_map_base * 16, k_size, stride=(1, 1), padding=(0, 0)),
            *_blocks(feature_map_base * 16, feature_map_base * 8, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 8, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 2, k_size, s_amount, p_amount),
            nn.ConvTranspose2d(feature_map_base * 2, img_channels, k_size, s_amount, p_amount),
            nn.Tanh()
        )

        self.img_size = img_size
        self.class_embedding = nn.Embedding(num_classes, embed_size)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Latent vector shape, N * z_dim * 1 * 1. Unsqueeze embedding in h and w dimension to concat with noise vector.
        """
        class_embedding = self.class_embedding(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, class_embedding], dim=1)  # Concat in c dimension.
        return self.G(z)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, image_size, image_channels):
        super(CustomImageDataset, self).__init__()
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        # class_name_to_idx = dict()
        # idx_to_class_name = dict()
        # for idx, class_name_folder in enumerate(self.class_list):
        #     class_name_to_idx[class_name_folder] = idx
        #     idx_to_class_name[idx] = class_name_folder

        self.image_labels_files_list = list()
        for idx, class_name_folder in enumerate(self.class_list):
            class_path = os.path.join(root_dir, class_name_folder)
            files_list = os.listdir(class_path)
            for image_file in files_list:
                self.image_labels_files_list.append(
                    (
                        os.path.join(class_path, image_file),
                        idx,
                    )
                )

        self.image_files_list_len = len(self.image_labels_files_list)
        # print(self.image_labels_files_list)
        # print(len(self.image_labels_files_list))

    def __len__(self):
        return self.image_files_list_len

    def __getitem__(self, idx):
        image_path, class_label = self.image_labels_files_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, class_label


class Trainer:
    def __init__(
            self,
            root_dir='',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            num_workers: int = 0,
            batch_size: int = 64,
            image_size: int = 64,
            image_channels: int = 3,
            num_epochs: int = 10000,
            z_dim: int = 100,
            learning_rate: float = 1e-4,
            generator_feature_map_base: int = 64,
            critic_feature_map_base: int = 64,
            critic_iterations: int = 5,
            lambda_gp: int = 10,
            num_classes: int = 10,
            gen_class_embed_size: int = 100,
    ) -> None:
        super(Trainer, self).__init__()

        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.CRITIC_ITERATIONS = critic_iterations
        self.LAMBDA_GP = lambda_gp
        self.BATCH_SIZE = batch_size
        self.Z_DIM = z_dim
        self.num_classes = num_classes
        self.gen_class_embed_size = gen_class_embed_size

        gan_dataset = CustomImageDataset(root_dir=root_dir, image_size=image_size, image_channels=image_channels)
        self.train_loader = DataLoader(
            gan_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        def _initialize_weights(model, mean=0.0, std=0.02):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                    nn.init.normal_(m.weight.data, mean=mean, std=std)

        self.G = Generator(
            z_dim=z_dim,
            img_channels=image_channels,
            feature_map_base=generator_feature_map_base,
            num_classes=num_classes,
            img_size=image_size,
            embed_size=gen_class_embed_size,
        ).to(self.device)
        self.C = Critic(
            img_channels=image_channels,
            feature_map_base=critic_feature_map_base,
            num_classes=num_classes,
            img_size=image_size,
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
        self.batch_size = batch_size
        self.fixed_noise = torch.randn((batch_size, z_dim, 1, 1), device=device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.step = 0

    def get_gradient_penalty(self, real, fake, labels):
        batch_size, c, h, w = real.shape
        epsilon = torch.rand((batch_size, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolated_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = self.C(interpolated_images, labels)
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

    def train(self):
        for epoch in range(self.NUM_EPOCHS):
            for batch_idx, (real, labels) in enumerate(self.train_loader):
                real = real.to(self.device)
                current_batch_size = real.shape[0]
                labels = labels.to(self.device)

                # Critic optimization.
                mean_iteration_critic_loss = 0
                for _ in range(self.CRITIC_ITERATIONS):
                    noise = torch.randn((current_batch_size, self.Z_DIM, 1, 1), device=self.device)
                    fake = self.G(noise, labels)
                    critic_real = self.C(real, labels).reshape(-1)
                    critic_fake = self.C(fake.detach(), labels).reshape(-1)
                    gradient_penalty = self.get_gradient_penalty(real=real, fake=fake, labels=labels)
                    loss_critic = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.LAMBDA_GP * gradient_penalty
                    )
                    mean_iteration_critic_loss += loss_critic.item() / self.CRITIC_ITERATIONS
                    self.opt_C.zero_grad()
                    loss_critic.backward()
                    self.opt_C.step()

                # Generator optimization.
                noise = torch.randn((current_batch_size, self.Z_DIM, 1, 1), device=self.device)
                fake = self.G(noise, labels)
                output = self.C(fake, labels).reshape(-1)
                loss_gen = -torch.mean(output)
                self.opt_G.zero_grad()
                loss_gen.backward()
                self.opt_G.step()

                if batch_idx % 10 == 0:
                    self.G.eval()
                    self.C.eval()
                    print(
                        f"EPOCH: [{epoch} / {self.NUM_EPOCHS}], BATCH: [{batch_idx} / {len(self.train_loader)}], "
                        f"LOSS(Mean) D: {mean_iteration_critic_loss:.4f}, LOSS G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise, labels)
                        img_grid_real = torchvision.utils.make_grid(real[:self.batch_size], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:self.batch_size], normalize=True)
                        self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)
                    self.step += 1
                    self.G.train(True)
                    self.C.train(True)


if __name__ == '__main__':
    trainer = Trainer(
        root_dir=r'dataset\mnist_sample',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    trainer.train()
