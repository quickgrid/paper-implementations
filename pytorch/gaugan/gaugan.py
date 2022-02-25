"""Pytorch GauGAN implementation.

Either segmentation one hot mask or rgb mask can be passed to discriminator with little modification.

Todo
    - Modify to try to generate and match mask also as loss.
    - Try discriminator with either segmentation image or label.
    - Use multiscale feature from discriminator to calculate loss.
    - Test conv bias, norm affine and other parameter effect on result.

References
    - https://arxiv.org/abs/1903.07291
    - https://keras.io/examples/generative/gaugan/
    - https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gaugan-series.md
    - https://github.com/NVlabs/SPADE
"""

import os
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision.transforms import transforms
from torchvision import models
from torch.nn.utils.spectral_norm import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional
from torch.backends import cudnn
from PIL import Image
from tqdm import tqdm


class LayerDebugger(nn.Module):
    def __init__(self) -> None:
        super(LayerDebugger, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class ImageEncoder(nn.Module):
    def __init__(
            self,
            img_size: int,
            latent_dim: int,
            apply_norm: bool = True,
            enable_dropout: bool = False,
            dropout_rate: float = 0.5,
    ) -> None:
        super(ImageEncoder, self).__init__()

        dropout_layer = list()
        if enable_dropout:
            dropout_layer = [nn.Dropout(p=dropout_rate)]

        def _get_block(
                _in_channels: int,
                _out_channels: int,
        ) -> list:
            norm_layer = list()
            if apply_norm:
                norm_layer = [nn.InstanceNorm2d(num_features=_out_channels, affine=False)]

            return [
                spectral_norm(nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False,
                )),
                *norm_layer,
                nn.LeakyReLU(negative_slope=0.2),
                *dropout_layer,
            ]

        channel_in = [3, 64, 128, 256, 512, 512]
        channel_out = [64, 128, 256, 512, 512, 512]
        linear_features = 8192

        conv_layers = list()
        for in_channels, out_channels in zip(channel_in, channel_out):
            conv_layers.extend(_get_block(_in_channels=in_channels, _out_channels=out_channels))

        self.encoder_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(
                in_features=((img_size // (2 ** len(channel_out))) ** 2) * channel_out[-1],
                out_features=linear_features
            ),
        )

        self.mean_out = nn.Linear(in_features=linear_features, out_features=latent_dim)
        self.variance_out = nn.Linear(in_features=linear_features, out_features=latent_dim)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder_layers(img)
        mean = self.mean_out(x)
        var = self.variance_out(x)
        return mean, var


class Discriminator(nn.Module):
    """Conv parameters are manually calculated using formula in pytorch Conv2D docs to keep output shape same.
    """

    def __init__(
            self,
            num_classes: int,
            device: torch.device,
            apply_norm: bool = True,
            enable_dropout: bool = False,
            dropout_rate: float = 0.5,
    ) -> None:
        super(Discriminator, self).__init__()

        dropout_layer = list()
        if enable_dropout:
            dropout_layer = [nn.Dropout(p=dropout_rate)]

        def _get_block(
                _in_channels: int, _out_channels: int, _stride: int, _padding: int, _dilation: int,
        ) -> nn.Sequential:
            norm_layer = list()
            if apply_norm:
                # norm_layer = [nn.BatchNorm2d(num_features=_out_channels, device=device)]
                norm_layer = [nn.InstanceNorm2d(num_features=_out_channels, affine=False, device=device)]

            return nn.Sequential(
                spectral_norm(nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=(4, 4),
                    padding=(_padding, _padding),
                    stride=(_stride, _stride),
                    dilation=(_dilation, _dilation),
                    device=device,
                    bias=False,
                )),
                *norm_layer,
                nn.LeakyReLU(negative_slope=0.2),
                *dropout_layer,
            )

        channel_in = [3 * 2, 64, 128, 256]
        # channel_in = [3 + num_classes, 64, 128, 256]
        channel_out = [64, 128, 256, 512]
        stride = [2, 2, 2, 1]
        padding = [3, 3, 3, 3]
        dilation = [2, 2, 2, 2]

        self.disc_multiscale_features = list()
        for in_channels, out_channels, stride, padding, dilation in zip(
                channel_in, channel_out, stride, padding, dilation
        ):
            self.disc_multiscale_features.append(
                _get_block(
                    _in_channels=in_channels,
                    _out_channels=out_channels,
                    _stride=stride,
                    _padding=padding,
                    _dilation=dilation,
                )
            )

        self.disc_out_layer = nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=(4, 4), padding=(3, 3), stride=(2, 2), dilation=(2, 2)
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, list]:
        x = torch.cat([img1, img2], dim=1)
        multiscale_features = list()
        for layer in self.disc_multiscale_features:
            x = layer(x)
            multiscale_features.append(x)
        x = self.disc_out_layer(x)
        multiscale_features.append(x)
        return x, multiscale_features


class SPADE(nn.Module):
    def __init__(
            self,
            out_channels: int,
            num_classes: int,
    ) -> None:
        super(SPADE, self).__init__()

        embed_dim = 128
        self.normalizer = nn.InstanceNorm2d(num_features=embed_dim, affine=False)

        self.embedding_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        self.gamma_conv = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        self.beta_conv = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, packed_tensor: torch.Tensor) -> torch.Tensor:
        prev_input, onehot_mask = packed_tensor
        mask = functional.interpolate(onehot_mask.float(), size=prev_input.shape[2:], mode='nearest')
        x = self.embedding_conv(mask)
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)
        var, mean = torch.var_mean(prev_input, dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(var + 1e-5)
        normalized = (prev_input - mean) / std
        output = gamma * normalized + beta
        return output


class SPADEResBlock(nn.Module):
    def __init__(
            self,
            in_filters: int,
            out_filters: int,
            num_classes: int,
            apply_spectral_norm: bool = True,
    ) -> None:
        super(SPADEResBlock, self).__init__()
        self.learned_skip = (in_filters != out_filters)
        min_filters = min(in_filters, out_filters)

        def _get_conv_layer(in_channels: int, out_channels: int, apply_bias: bool = True) -> list:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=apply_bias,
            )
            if apply_spectral_norm:
                conv_layer = spectral_norm(conv_layer)
                return [conv_layer]
            return [conv_layer]

        self.spade_res_block_1 = nn.Sequential(
            SPADE(num_classes=num_classes, out_channels=in_filters),
            nn.LeakyReLU(negative_slope=0.2),
            *_get_conv_layer(in_channels=in_filters, out_channels=min_filters),
        )

        self.spade_res_block_2 = nn.Sequential(
            SPADE(out_channels=min_filters, num_classes=num_classes),
            nn.LeakyReLU(negative_slope=0.2),
            *_get_conv_layer(in_channels=min_filters, out_channels=out_filters),
        )

        self.learned_skip_path = nn.Sequential(
            SPADE(out_channels=in_filters, num_classes=num_classes),
            nn.LeakyReLU(negative_slope=0.2),
            *_get_conv_layer(in_channels=in_filters, out_channels=out_filters, apply_bias=False),
        )

    def forward(self, packed_tensor: torch.Tensor) -> torch.Tensor:
        x, onehot_mask = packed_tensor
        x_skip = x
        x = self.spade_res_block_1((x, onehot_mask))
        x = self.spade_res_block_2((x, onehot_mask))
        if self.learned_skip:
            x_skip = self.learned_skip_path((x_skip, onehot_mask))
        x = x + x_skip
        return x


class GaussianSampler(nn.Module):
    def __init__(
            self,
            batch_size: int,
            latent_dim: int,
            device: torch.device,
    ) -> None:
        super(GaussianSampler, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, variance = x
        std = torch.exp(0.5 * variance)
        epsilon = torch.randn_like(std)
        noise_input = std * epsilon + mean
        return noise_input


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            num_classes: int,
            device: torch.device,
    ) -> None:
        super(Generator, self).__init__()

        def _get_res_block(_in_filters: int, _out_filters: int) -> nn.Sequential:
            return nn.Sequential(
                SPADEResBlock(
                    in_filters=_in_filters, out_filters=_out_filters, num_classes=num_classes
                ).to(device=device),
                nn.Upsample(scale_factor=(2, 2)),
            )

        self.initial_shape = 1024

        filter_list = [self.initial_shape, 1024, 1024, 512, 256, 128, 64]
        self.filter_list_len = len(filter_list)

        self.generator_middle_layers = list()
        for i in range(self.filter_list_len - 1):
            self.generator_middle_layers.append(
                _get_res_block(_in_filters=filter_list[i], _out_filters=filter_list[i+1])
            )

        self.generator_input_layers = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128 * 128),
        )

        # Change conv layer stride for custom image size.
        self.generator_output_layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=filter_list[-1], out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, latent_vector: torch.Tensor, onehot_mask: torch.Tensor) -> torch.Tensor:
        x = self.generator_input_layers(latent_vector)
        x = x.view(-1, self.initial_shape, 4, 4)
        for mid_layer in self.generator_middle_layers:
            x = mid_layer((x, onehot_mask))
        x = self.generator_output_layers(x)
        return x


class VggLoss(nn.Module):
    """Use vgg intermediate layers to calculate perceptual loss.
    """

    def __init__(
            self,
            device: torch.device,
            debug: bool = False,
    ) -> None:
        super(VggLoss, self).__init__()
        model = models.vgg19(pretrained=True).to(device=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if debug:
            print(model.features)
            train_nodes, eval_nodes = get_graph_node_names(model)
            print('train_nodes')
            print(train_nodes)
            print('eval_nodes')
            print(eval_nodes)

        return_nodes = {
            'features.1': 'out_0',
            'features.6': 'out_1',
            'features.11': 'out_2',
            'features.20': 'out_3',
            'features.29': 'out_4',
        }
        self.feature_count = len(return_nodes)

        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes=return_nodes
        )

        self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_out = self.feature_extractor(x)
        y_out = self.feature_extractor(y)
        loss = 0.0
        for i in range(self.feature_count):
            loss += self.layer_weights[i] * functional.l1_loss(x_out[f'out_{i}'], y_out[f'out_{i}'])
        return loss


class GauganDataset(Dataset):
    """Real images should be jpg and Segmentation image should be in png format.
    """

    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int,
            num_classes: int,
    ) -> None:
        super(GauganDataset, self).__init__()

        self.num_classes = num_classes + 1

        self.root_dir = root_dir
        self.image_labels_files_list = list()
        for root, dirs, files in os.walk(root_dir):
            for names in files:
                if names.endswith('.jpg'):
                    base_name = names.split('.')[0]
                    self.image_labels_files_list.append(
                        (
                            os.path.join(root, f'{base_name}.jpg'),
                            os.path.join(root, f'{base_name}.png'),
                        )
                    )

        self.image_files_list_len = len(self.image_labels_files_list)

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        self.segmentation_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        self.segmentation_label_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return self.image_files_list_len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, segmentation_path = self.image_labels_files_list[idx]

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.img_transform(image)

        segmentation_image_original = Image.open(segmentation_path)

        segmentation_image = segmentation_image_original.convert('RGB')
        segmentation_image = self.segmentation_transform(segmentation_image)

        segmentation_label = segmentation_image_original.convert('P')
        segmentation_label = self.segmentation_label_transform(segmentation_label)
        segmentation_label = functional.one_hot(segmentation_label.long(), num_classes=self.num_classes)
        segmentation_label = torch.permute(segmentation_label.squeeze(), (2, 0, 1))

        return image, segmentation_image, segmentation_label


def feature_matching_loss(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> float:
    pred_count_weight = 1 / len(real_preds)
    _feature_matching_loss = 0.0
    for real_features, fake_features in zip(real_preds, fake_preds):
        _feature_matching_loss += functional.l1_loss(real_features, fake_features) * pred_count_weight
    return _feature_matching_loss


class Trainer:
    def __init__(
            self,
            num_classes: int,
            root_dir='',
            device: str = None,
            checkpoint_path: str = None,
            save_checkpoint_every: int = 20,
            num_workers: int = 0,
            batch_size: int = 3,
            image_size: int = 256,
            image_channels: int = 3,
            num_epochs: int = 10000,
            latent_dim: int = 256,
            gen_learning_rate: float = 0.0001,
            disc_learning_rate: float = 0.0004,
            disc_iterations: int = 1,
            debug: bool = False,
    ) -> None:

        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.profiler.profile(enabled=False)
        cudnn.benchmark = True

        if debug:
            torch.autograd.set_detect_anomaly(True)
            torch.autograd.profiler.emit_nvtx(enabled=True)
            torch.autograd.profiler.profile(enabled=True)
            cudnn.benchmark = False

        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_every = save_checkpoint_every
        self.disc_iterations = disc_iterations

        gan_dataset = GauganDataset(
            root_dir=root_dir,
            image_size=image_size,
            image_channels=image_channels,
            num_classes=num_classes,
        )
        self.train_loader = DataLoader(
            gan_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self.vgg_model = VggLoss(device=self.device)

        self.image_encoder = ImageEncoder(img_size=image_size, latent_dim=latent_dim)
        self.noise_sampler = GaussianSampler(batch_size=batch_size, latent_dim=latent_dim, device=self.device)
        self.generator = Generator(latent_dim=latent_dim, num_classes=num_classes + 1, device=self.device)
        self.discriminator = Discriminator(num_classes=num_classes + 1, device=self.device)

        self.image_encoder.to(device=self.device)
        self.noise_sampler.to(device=self.device)
        self.generator.to(device=self.device)
        self.discriminator.to(device=self.device)

        def _initialize_weights(model, mean=0.0, std=0.02):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    # nn.init.normal_(m.weight.data, mean=mean, std=std)
                    nn.init.xavier_normal_(m.weight.data)
                    # nn.init.kaiming_normal_(m.weight.data)

        _initialize_weights(self.image_encoder)
        _initialize_weights(self.generator)
        _initialize_weights(self.discriminator)

        encoder_generator_parameters = list(self.generator.parameters()) + list(self.image_encoder.parameters())
        self.gen_optimizer = optim.Adam(
            params=encoder_generator_parameters, lr=gen_learning_rate, betas=(0.0, 0.999)
        )

        self.disc_optimizer = optim.Adam(
            params=self.discriminator.parameters(), lr=disc_learning_rate, betas=(0.0, 0.999)
        )

        self.fixed_noise = torch.randn((batch_size, latent_dim), device=self.device)

        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.step = 0

        self.start_epoch = 0
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            with tqdm(self.train_loader) as tqdm_train_loader:
                for batch_idx, (real_image, segmentation_image, segmentation_label) in enumerate(tqdm_train_loader):
                    real_image = real_image.to(self.device)
                    segmentation_image = segmentation_image.to(self.device)
                    segmentation_label = segmentation_label.to(self.device)

                    # Train discriminator.
                    for i in range(self.disc_iterations):
                        mean, var = self.image_encoder(real_image)
                        latent_vector = self.noise_sampler((mean, var))
                        generated_image = self.generator(latent_vector, segmentation_label)

                        fake_pred, fake_pred_multiscale_features = self.discriminator(
                            segmentation_image, generated_image
                        )
                        real_pred, real_pred_multiscale_features = self.discriminator(
                            segmentation_image, real_image
                        )

                        fake_pred = fake_pred.reshape(-1)
                        real_pred = real_pred.reshape(-1)

                        loss_real = -torch.mean(
                            torch.min(real_pred - 1, torch.zeros_like(real_pred, requires_grad=False))
                        )
                        loss_fake = -torch.mean(
                            torch.min(-fake_pred.detach() - 1, torch.zeros_like(fake_pred, requires_grad=False))
                        )
                        discriminator_loss = (loss_fake + loss_real) * 0.5

                        self.disc_optimizer.zero_grad(set_to_none=True)
                        discriminator_loss.backward()
                        self.disc_optimizer.step()

                    # Train generator
                    fake_pred, fake_pred_multiscale_features = self.discriminator(segmentation_image, generated_image)
                    real_pred, real_pred_multiscale_features = self.discriminator(segmentation_image, real_image)
                    fake_pred = fake_pred.reshape(-1)

                    loss_gen = -torch.mean(fake_pred)
                    loss_kldiv = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
                    loss_vgg = self.vgg_model(real_image, generated_image)
                    loss_features = feature_matching_loss(
                        real_pred_multiscale_features,
                        fake_pred_multiscale_features,
                    )

                    generator_loss = loss_gen + 0.05 * loss_kldiv + 10 * loss_vgg + 10 * loss_features

                    self.gen_optimizer.zero_grad(set_to_none=True)
                    generator_loss.backward()
                    self.gen_optimizer.step()

                    tqdm_train_loader.set_description(
                        f'LOSS, disc: {discriminator_loss:.2f}, '
                        f'generator: {generator_loss:.2f}, '
                        f'gan: {loss_gen:.2f}, '
                        f'kl: {loss_kldiv:.2f}, '
                        f'vgg: {loss_vgg:.2f}, '
                        f'features: {loss_features:.2f}'
                    )

                    if batch_idx % self.save_every == self.save_every - 1:
                        self.generator.eval()
                        self.discriminator.eval()

                        with torch.no_grad():
                            fake = self.generator(self.fixed_noise, segmentation_label)
                            img_grid_real = torchvision.utils.make_grid(real_image[:self.batch_size], normalize=True)
                            img_grid_fake = torchvision.utils.make_grid(fake[:self.batch_size], normalize=True)
                            self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                            self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)
                            self.step += 1

                            torch.save({
                                'epoch': epoch,
                                'generator_state_dict': self.generator.state_dict(),
                                'discriminator_state_dict': self.discriminator.state_dict(),
                                'generator_optimizer_state_dict': self.gen_optimizer.state_dict(),
                                'discriminator_optimizer_state_dict': self.disc_optimizer.state_dict(),
                            }, f'checkpoints/checkpoint_{epoch}.pt')

                        self.discriminator.train()
                        self.generator.train()


if __name__ == '__main__':
    trainer = Trainer(
        root_dir=r'C:\staging\gaugan_data\base',
        num_classes=12,
        # checkpoint_path='checkpoints/checkpoint_6.pt'
    )
    trainer.train()
