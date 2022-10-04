"""ProGAN Implementation.

References:
    - Code based on, https://www.youtube.com/watch?v=nkQHASviYac.
    - Code, https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/ProGAN.
    - ProGAN paper, https://arxiv.org/abs/1710.10196.
"""
import os
import pathlib
import random
from datetime import datetime
from typing import Tuple, List
from math import log2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.backends import cudnn


class Config:
    """Model configurations.

    layer_channel_scale_factors: Scale factor of channels/feature maps based on given max channel size.
        On paper for given size the feature map per block is scaled by, [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32].

    non_initial_block_count: One less than the number of blocks in generator or discriminator. Both gen and disc
        has same number of blocks mirrored in reverse. This is equal to `layer_channel_scale_factors` the channel or
        feature map scaling list size. One less because the first and last blocks of gen and disc are forwarded
        separately instead of in loops like other blocks.
    """
    layer_channel_scale_factors: List[float] = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    non_initial_block_count: int = len(layer_channel_scale_factors) - 1


class WSConv2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (3, 3),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (1, 1),
            gain: int = 2
    ):
        """Weight scaled convolution.

        Look into ProGAN paper, section 4.1 for weight initialization explanation.
        Weights are initialized from a normal distribution with mean 0, variance 1 and scaled at runtime with equation,
        `w_hat_i = w_i / c`.

        Kaiming he normal equation, https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_.
        `std = gain / sqrt(fan_mode)`, using, `fan_mode = fan_in`, number of input units should be the number
        of channels multiplied by height, width. In this case (height, width) is (kernel_size, kernel_size).
        So, `fan_in = kernel_size * kernel_size * in_channels`.

        Further details on bias, weight initialization application found in section A.1.

        Args:
            gain: Scaling factor for weight initialization.

        Returns:
            Scaled convolution.
        """
        super(WSConv2D, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.scale = (gain / (in_channels * kernel_size[0] ** 2)) ** 0.5

        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv.bias = None

        # Initialize conv layer parameter
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.scale) + self.bias


class WSConvTranspose2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (4, 4),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            gain: int = 2
    ):
        super(WSConvTranspose2d, self).__init__()

        self.conv_t = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.scale = (gain / (in_channels * kernel_size[0] ** 2)) ** 0.5

        bias = self.conv_t.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv_t.bias = None

        # Initialize conv layer parameter.
        nn.init.normal_(self.conv_t.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_t(x * self.scale) + self.bias


class PixelNorm(nn.Module):
    def __init__(self):
        """Pixel wise feature vector normalization for generator.

        Look into ProGAN paper, section 4.2 for equation explanation and epsilon value.
        Normalization is performed depth wise for each feature map.

        Returns:
            pixel wise normalized feature vector
        """
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_pixel_norm: bool = True):
        """Double conv blocks as shown in table 2 of paper.

        Pixel norm is only applied in generator after each weight scaled conv blocks as referenced in section A.1.
        """
        super(DoubleConvBlock, self).__init__()
        self.conv1 = WSConv2D(in_channels, out_channels)
        self.conv2 = WSConv2D(out_channels, out_channels)
        self.use_pixel_norm = use_pixel_norm
        self.leaky = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky(self.conv1(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x
        x = self.leaky(self.conv2(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim: int, base_channels: int, img_channels: int = 3):
        """ProGAN Generator.

        Pixel norm application details, LeakyRelu with value is mentioned in section A.1 in ProGAN paper.
        Pixel norm is applied after each 3x3 convolution as mentioned in the paper.

        The network architecture is available in Table 2. Here, initial is the first block in generator
        diagram as it is different from architecture of other blocks.

        Figure 2, detail part `toRGB`, `fromRGB` details. Both use 1x1 convolutions.
        `toRGB` projects feature map to RGB 3 channels and `fromRGB` does the opposite from 3 to given channels.
        """

        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            WSConvTranspose2d(in_channels=z_dim, out_channels=base_channels),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WSConv2D(in_channels=base_channels, out_channels=base_channels),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.progressive_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()

        self.initial_to_rgb_layers = nn.Sequential(
            WSConv2D(
                in_channels=base_channels, out_channels=img_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            ),
            PixelNorm(),
        )
        self.to_rgb_layers.append(self.initial_to_rgb_layers)

        for i in range(Config.non_initial_block_count):
            conv_in_c = int(base_channels * Config.layer_channel_scale_factors[i])
            conv_out_c = int(base_channels * Config.layer_channel_scale_factors[i + 1])
            self.progressive_blocks.append(DoubleConvBlock(conv_in_c, conv_out_c))
            self.to_rgb_layers.append(
                nn.Sequential(
                    WSConv2D(
                        in_channels=conv_out_c,
                        out_channels=img_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0)
                    ),
                    PixelNorm(),
                )
            )

    @staticmethod
    def fade_in(alpha: float, upscaled: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """Section A.1 mentions training and generated images are represented in [-1, 1].

        Tanh activation is used to output [-1, 1].
        Figure 2, diagram (b) provides equation residual block like sum.

        Args:
            alpha: Fading value between current layer output and previous layer upsampled image. Values from 0 to 1.
            upscaled: Upsampled image of previous layer.
            generated: Output of current layer.

        Returns:
            Faded data between current layer output resolution and previous layer output upsampled.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x: torch.Tensor, alpha: float, current_block_count: int) -> torch.Tensor:
        """Upsample 2x and pass to next progressive layer.

        Look into Table 2 and figure 2 diagram for building generator model.

        Args:
            x:
            alpha: Layer fading value.
            current_block_count:
        """
        out = self.initial(x)

        if current_block_count == 0:
            return self.initial_to_rgb_layers(out)

        upscaled = None
        for block in range(current_block_count):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest-exact')
            out = self.progressive_blocks[block](upscaled)

        final_upscaled = self.to_rgb_layers[current_block_count - 1](upscaled)
        final_output = self.to_rgb_layers[current_block_count](out)

        return self.fade_in(upscaled=final_upscaled, generated=final_output, alpha=alpha)


class Discriminator(nn.Module):
    def __init__(self, base_channels: int, img_channels: int = 3):
        """Discriminator, or critic if wgan-gp loss is used.

        Architecture is opposite of generator.
        """
        super(Discriminator, self).__init__()

        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(Config.non_initial_block_count, 0, -1):
            conv_in_c = int(base_channels * Config.layer_channel_scale_factors[i])
            conv_out_c = int(base_channels * Config.layer_channel_scale_factors[i - 1])
            self.progressive_blocks.append(
                DoubleConvBlock(in_channels=conv_in_c, out_channels=conv_out_c, use_pixel_norm=False)
            )
            self.from_rgb_layers.append(
                WSConv2D(
                    in_channels=img_channels,
                    out_channels=conv_in_c,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                )
            )

        # For 4x4 resolution.
        self.final_from_rgb_layers = WSConv2D(
            in_channels=img_channels, out_channels=base_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.from_rgb_layers.append(self.final_from_rgb_layers)

        self.final_block = nn.Sequential(
            WSConv2D(
                in_channels=base_channels + 1,
                out_channels=base_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1)
            ),
            nn.LeakyReLU(0.2),
            WSConv2D(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=(4, 4),
                padding=(0, 0),
                stride=(1, 1)
            ),
            nn.LeakyReLU(0.2),
            WSConv2D(
                in_channels=base_channels,
                out_channels=1,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1)
            ),
        )

    @staticmethod
    def fade_in(alpha: float, downscaled: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return alpha * out + (1 - alpha) * downscaled

    @staticmethod
    def minibatch_std(x: torch.Tensor) -> torch.Tensor:
        """Section 3, paragraph 1, feature statistics calculated along minibatch and merged along channel dimension.
        """
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(
            self,
            x: torch.Tensor,
            alpha: float,
            current_block_count: int
    ) -> torch.Tensor:
        """As in section 3, minibatch statistics is concatenated layer output toward end of discriminator.
        """
        current_block = len(self.progressive_blocks) - current_block_count
        out = self.leaky(self.from_rgb_layers[current_block](x))

        if current_block_count == 0:
            out = Discriminator.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.from_rgb_layers[current_block + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progressive_blocks[current_block](out))

        out = self.fade_in(downscaled=downscaled, out=out, alpha=alpha)

        for block in range(current_block + 1, len(self.progressive_blocks)):
            out = self.progressive_blocks[block](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


class CustomImageClassDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int
    ):
        super(CustomImageClassDataset, self).__init__()
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

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

    def __len__(self) -> int:
        return self.image_files_list_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, class_label = self.image_labels_files_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transform(image), class_label


class Tester:
    def __init__(self, device: str = 'cuda'):
        super(Tester, self).__init__()
        self.device = device

    def test_all(self) -> None:
        self.test_model_architecture()

    def test_wsconv(self) -> None:
        x = torch.randn(size=(4, 3, 32, 32), device=self.device)
        wsconv = WSConv2D(in_channels=3, out_channels=128).to(self.device)
        out = wsconv(x)
        print(out.shape)

    def test_double_conv(self) -> None:
        x = torch.randn(size=(4, 64, 32, 32), device=self.device)
        double_conv = DoubleConvBlock(in_channels=64, out_channels=128).to(self.device)
        out = double_conv(x)
        print(out.shape)

    def test_model_architecture(self) -> None:
        z_dim = 256
        base_channels = 128
        gen = Generator(z_dim, base_channels, img_channels=3).to(self.device)
        critic = Discriminator(base_channels, img_channels=3).to(self.device)

        print(gen)
        print(critic)

        for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            num_blocks = int(log2(img_size / 4))
            print(f'Res: {img_size}, Blocks: {num_blocks}')
            noise = torch.randn((1, z_dim, 1, 1), device=self.device)
            img = gen(noise, alpha=0.5, current_block_count=num_blocks)
            assert img.shape == (1, 3, img_size, img_size)
            critic_score = critic(img, alpha=0.5, current_block_count=num_blocks)
            print(critic_score.shape)
            assert critic_score.shape == (1, 1), 'Error output do not match.'
            print(f"Success! At img size: {img_size}")


class Utils:
    @staticmethod
    def collate_fn(batch):
        """Discard none samples.
        """
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    @staticmethod
    def save_checkpoint(
            epoch: int,
            model: nn.Module,
            filename: str,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> None:
        checkpoint_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer:
            checkpoint_dict['optimizer'] = optimizer.state_dict()
        if scheduler:
            checkpoint_dict['scheduler'] = scheduler.state_dict()
        if scheduler:
            checkpoint_dict['grad_scaler'] = grad_scaler.state_dict()

        torch.save(checkpoint_dict, filename)

    @staticmethod
    def load_checkpoint(
            model: nn.Module,
            filename: str,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> int:
        saved_model = torch.load(filename, map_location="cuda")
        model.load_state_dict(saved_model['state_dict'], strict=False)
        if 'optimizer' in saved_model:
            optimizer.load_state_dict(saved_model['optimizer'])
        if 'scheduler' in saved_model:
            scheduler.load_state_dict(saved_model['scheduler'])
        if 'grad_scaler' in saved_model:
            grad_scaler.load_state_dict(saved_model['grad_scaler'])
        return saved_model['epoch']

    @staticmethod
    def train_mode(debug: bool, seed: int = 42) -> None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if debug:
            torch.autograd.set_detect_anomaly(True)
            torch.autograd.profiler.emit_nvtx(enabled=True)
            torch.autograd.profiler.profile(enabled=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.emit_nvtx(enabled=False)
            torch.autograd.profiler.profile(enabled=False)
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def gradient_penalty(critic, real, fake, alpha, current_block_count, device):
        batch_size, c, h, w = real.shape
        epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
        interpolated_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = critic(interpolated_images, alpha, current_block_count)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=False,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty


class Trainer:
    def __init__(
            self,
            root_dir: str = '',
            device: str = 'cuda',
            checkpoint_path: str = '',
            save_checkpoint_every: int = 20,
            num_workers: int = 0,
            batch_size: int = 16,
            image_size: int = 256,
            image_channels: int = 3,
            num_epochs: int = 10000,
            latent_dim: int = 256,
            max_feature_map_size: int = 128,  # In progan paper 512 is used.
            learning_rate: float = 0.001,
            debug: bool = False,
    ):
        Utils.train_mode(debug=debug)

        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.save_every = save_checkpoint_every

        gan_dataset = CustomImageClassDataset(
            root_dir=root_dir,
            image_size=image_size,
            image_channels=image_channels,
        )
        self.train_loader = DataLoader(
            gan_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=Utils.collate_fn
        )

        self.gen = Generator(
            z_dim=latent_dim, base_channels=max_feature_map_size, img_channels=image_channels,
        ).to(self.device)

        self.disc = Discriminator(
            base_channels=max_feature_map_size, img_channels=image_channels,
        ).to(self.device)

        self.optimizer_gen = optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
        self.optimizer_disc = optim.Adam(self.disc.parameters(), lr=learning_rate, betas=(0.0, 0.99))

        # Tensorboard settings.
        current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer_real = SummaryWriter(f'logs/real/{current_datetime}/')
        self.writer_fake = SummaryWriter(f'logs/fake/{current_datetime}/')
        self.tensorboard_step = 0

        # Model loading for training resumption.
        self.start_epoch = 0
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True)
        if checkpoint_path is not None:
            gen_epoch_num = Utils.load_checkpoint(
                model=self.gen,
                optimizer=self.optimizer_gen,
                filename=checkpoint_path,
            )
            disc_epoch_num = Utils.load_checkpoint(
                model=self.gen,
                optimizer=self.optimizer_gen,
                filename=checkpoint_path,
            )

            assert gen_epoch_num == disc_epoch_num, 'Training epoch of generator and discrimator model do not match.'
            self.start_epoch = gen_epoch_num

        # Set both generator and discriminator in train mode.
        self.gen.train()
        self.disc.train()

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            for batch_idx, (real) in enumerate(self.train_loader):
                real = real.to(self.device)
                current_batch_size = real.shape[0]


if __name__ == "__main__":
    tester = Tester()
    # tester.test_all()
    # tester.test_wsconv()
    # tester.test_double_conv()
    tester.test_model_architecture()
