"""DCGAN Implementation with least squares loss.

Some comments useful for understanding code are deleted. They can be found on main file.
For generator loss any of BCE loss or MSE loss can be used as they both generates result.

TODO:
    - Bring code under latest formatting.

References:
    - Code implementation tutorial, https://www.youtube.com/watch?v=IZtv9s_Wx9I.
    - DCGAN paper, https://arxiv.org/pdf/1511.06434.pdf.
    - Least Squares GAN (LSGAN) paper, https://arxiv.org/pdf/1611.04076.pdf.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.nn import functional
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    """Critic class of DCGAN.

    Attributes:
        D: Sequential discriminator model for classifying 64 x 64 images to real or fake label.
    """

    def __init__(self, img_channels, d_channels):
        """Init discriminator variables.

        Section 3 of paper suggest batch normalization is applied to every layer except input layer in discriminator.
        Also fully connected layers are removed instead convolutional features in last layer is flattened then fed into
        single sigmoid output.
        It also mentions pooling is not used instead strided convolution is used.

        Args:
            img_channels: Number of channels in dataset image. For example, 3 for RGB, 1 for greyscale.
            d_channels: Output feature map count which is increased every layer.
        """
        super(Discriminator, self).__init__()

        def _blocks(in_channels, out_channels, kernel_size, stride, padding):
            """Combined block of strided convolution, batch normalization and leaky relu activation function.

            Section 4 of DCGAN paper mentions to use LeakyReLU with slope of leak to 0.2.

            Returns:
                A combined sequential block of Conv2d, BatchNorm2d, LeakyReLU.
            """
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        self.D = nn.Sequential(
            # For input shape, (BATCH_SIZE = N, C = 3, H = 64, H = 64).
            nn.Conv2d(img_channels, d_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            _blocks(d_channels, d_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            _blocks(d_channels * 2, d_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            _blocks(d_channels * 4, d_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(d_channels * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Run forward pass on discriminator to get real or fake label.

        Args:
            x: Input image.

        Returns:
            Critic prediction whether input is real or fake.
        """
        return self.D(x)


class Generator(nn.Module):
    """Generator class for DCGAN.

    Attributes:
        G: Generator model for generating image with `C` channels from noise vector `Z`.
    """

    def __init__(self, z_dim, img_channels, g_channels):
        """Initialize generator structure.

        In paper section 3 mentions to not apply Batch normalization to output layer.
        The architecture for generator is shown in Figure 1.
        Tanh activation function is applied in generator output layer and ReLU is applied for all other layers as
        noted in section 3 and 4. As `Tanh` outputs values between `(-1, 1)` the input images are scaled as such.
        """
        super(Generator, self).__init__()

        def _block(in_channels, out_channels, kernel_size, stride, padding):
            """Combined block of fractionally strided convolution, batch normalization and leaky relu activation.

            In pytorch, fractionally strided convolution/tenspose convolution is implemented with `ConvTranspose2d`.
            Batch normalization is applied on image channels so here after transpose convolution is has `out_channels`.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.G = nn.Sequential(
            # For input shape, (BATCH_SIZE = N, C = z_dim, H = 1, H = 1).
            _block(z_dim, g_channels * 16, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            _block(g_channels * 16, g_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            _block(g_channels * 8, g_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            _block(g_channels * 4, g_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ConvTranspose2d(g_channels * 2, img_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, z):
        """Run forward pass on generator to generate images from noise.

        Args:
            z: Noise vector.

        Returns:
            Generated Images.
        """
        return self.G(z)


class Trainer:
    """Trainer class to train DCGAN model.

    It has dataset loading, transformation, weight initialization and training.
    """

    def __init__(
            self
    ):
        super(Trainer, self).__init__()

        # Setup training variables.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LEARNING_RATE = 2e-4
        self.BATCH_SIZE = 128
        self.IMAGE_SIZE = 64
        self.IMG_CHANNELS = 3
        self.Z_DIM = 100
        self.G_CHANNELS = 64
        self.D_CHANNELS = 64
        self.NUM_EPOCHS = 10000

        # Setup data transform.
        self.transform_config = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(self.IMG_CHANNELS)],
                std=[0.5 for _ in range(self.IMG_CHANNELS)]
            )
        ])

        # Load dataset.
        # self.dataset = datasets.MNIST(root='dataset/', train=True, transform=self.transform_config, download=True)
        self.dataset = datasets.ImageFolder(root=r'C:\staging\celeba_hq', transform=self.transform_config)
        self.loader = DataLoader(self.dataset, batch_size=self.BATCH_SIZE, shuffle=True, pin_memory=True)

        # Create generator and discriminator models with weights from paper.
        self.G = Generator(self.Z_DIM, self.IMG_CHANNELS, self.G_CHANNELS).to(self.device)
        self.D = Discriminator(self.IMG_CHANNELS, self.D_CHANNELS).to(self.device)

        def initialize_weights(model):
            """Initialize weights of the model.

            Section 4 of paper mentions that all weights are initialized from zero centered normal distribution with
            standard deviation of 0.02.

            Args:
                model: Generator or Critic model for which layer weights are initialized.
            """
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inplace weight assign.

        initialize_weights(self.G)
        initialize_weights(self.D)

        # Setup optimizer and loss function.
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.LEARNING_RATE, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.LEARNING_RATE, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss().to(self.device)

        # Setup visualization for tensorboard.
        self.fixed_noise = torch.randn(32, self.Z_DIM, 1, 1).to(self.device)
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.writer_real = SummaryWriter(f"logs/real")
        self.step = 0

    def train(self):
        # Set both models to train mode
        self.G.train(True)
        self.D.train(True)

        # Run training loop.
        for epoch in range(self.NUM_EPOCHS):
            for batch_idx, (real, _) in enumerate(self.loader):
                real = real.to(self.device)
                noise = torch.randn((self.BATCH_SIZE, self.Z_DIM, 1, 1), device=self.device)
                fake = self.G(noise)

                # Train discriminator, `max log(D(x)) + log(1 - D(G(z)))`.
                d_real = self.D(real).squeeze()
                # loss_d_real = self.criterion(d_real, torch.ones_like(d_real))
                loss_d_real = functional.mse_loss(d_real, torch.ones_like(d_real))
                d_fake = self.D(fake.detach()).squeeze()
                # loss_d_fake = self.criterion(d_fake, torch.zeros_like(d_fake))
                loss_d_fake = functional.mse_loss(d_fake, torch.zeros_like(d_fake))

                loss_d = (loss_d_real + loss_d_fake) * 0.5
                self.D.zero_grad()
                loss_d.backward()
                self.opt_D.step()

                # Train generator, `min log(1 - D(G(z)))` <=> `max log(D(G(z)))`. Look into Goodfellow, GAN paper.
                d_fake = self.D(fake).squeeze()
                # loss_g = self.criterion(d_fake, torch.ones_like(d_fake))
                loss_g = functional.mse_loss(d_fake, torch.ones_like(d_fake))
                self.G.zero_grad()
                loss_g.backward()
                self.opt_G.step()

                # Print loss and show images in tensorboard.
                if batch_idx % 50 == 0:
                    print(
                        f"EPOCH: [{epoch} / {self.NUM_EPOCHS}], BATCH: [{batch_idx} / {len(self.loader)}], "
                        f"LOSS D: {loss_d:.4f}, LOSS G: {loss_g:.4f}"
                    )
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                        self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)
                self.step += 1


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
