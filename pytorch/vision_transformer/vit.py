"""Implementation of Vision Transformer (ViT) in pytorch.

Notes
    - Add learnable class token embedding.
    - Remove `drop_last` for smaller datasets.
    - Add accuracy logic print and in tensorboard.
    - Add loss data to tensorboard.

References
    - https://keras.io/examples/vision/image_classification_with_vision_transformer/.
    - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py.
    - https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py.
"""

import os
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


class VisionTransformerDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int,
    ) -> None:
        super(VisionTransformerDataset, self).__init__()
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)
        print(self.class_list)

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

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        image_path, class_label = self.image_labels_files_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, class_label


class TransformerEncoderModel(nn.Module):
    def __init__(
            self,
            num_heads: int,
            embedding_dim: int,
            mlp_hidden_dim: int = 2048,
            mlp_dropout: float = 0.0,
    ) -> None:
        super(TransformerEncoderModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=0.1)
        # self.layer_norm = nn.LayerNorm(embedding_dim)
        self.layer_norm = nn.InstanceNorm1d(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_hidden_dim, out_features=embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = self.layer_norm(x)
        attn_output, attn_output_weights = self.multihead_attn(query=x, key=x, value=x)
        x_skip = attn_output + x_skip
        x = self.layer_norm(x_skip)
        x = self.mlp(x) + x_skip
        return x


class PatchPositionEncoder(nn.Module):
    def __init__(
            self,
            num_patches: int,
            embedding_dim: int,
            device: torch.device,
    ) -> None:
        super(PatchPositionEncoder, self).__init__()
        # self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.device = device

        # self.num_patches = num_patches + 1  # Remove +1 to remove class token.
        self.num_patches = num_patches
        self.patch_embedding = nn.LazyLinear(out_features=embedding_dim)
        self.position_embedding = nn.Embedding(num_embeddings=num_patches, embedding_dim=embedding_dim)
        self.positions = torch.arange(start=0, end=self.num_patches, step=1, dtype=torch.int, device=self.device)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        patch_e = self.patch_embedding(patch)
        # patch_e = torch.cat([self.class_token, patch_e], dim=1)     # Disable this for removing class token.
        pos_e = self.position_embedding(self.positions)
        encoding = patch_e + pos_e
        return encoding


class ClassificationMLP(nn.Module):
    def __init__(
            self,
            num_classes: int,
            embedding_dim: int,
            patch_count: int,
    ) -> None:
        super(ClassificationMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.class_mapping_head = nn.Linear(in_features=embedding_dim * patch_count, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.class_mapping_head(x)
        return x


class VisionTransformerModel(nn.Module):
    def __init__(
            self,
            num_heads: int,
            embedding_dim: int,
            transformer_layers_count: int,
            num_patches: int,
            device: torch.device,
    ) -> None:
        super(VisionTransformerModel, self).__init__()

        self.transformer_layers_count = transformer_layers_count
        self.transformer_layers = nn.ModuleList()

        for _ in range(transformer_layers_count):
            self.transformer_layers.append(
                TransformerEncoderModel(num_heads=num_heads, embedding_dim=embedding_dim),
            )

        self.patch_encoder = PatchPositionEncoder(
            num_patches=num_patches,
            embedding_dim=embedding_dim,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_encoder(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        return x


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            checkpoint_path: str = None,
            device: str = None,
            num_epochs: int = 1000,
            batch_size: int = 64,
            save_every: int = 20,
            num_workers: int = 4,
            image_size: int = 72,
            patch_size: int = 6,
            learning_rate: float = 0.001,
            patch_embedding_dim: int = 48,
            transformer_layers_count: int = 5,
            num_heads: int = 4,
            image_channels: int = 3,
    ) -> None:
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_every = save_every
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.image_channels = image_channels
        self.num_heads = num_heads
        self.patch_embedding_dim = patch_embedding_dim
        self.transformer_layers_count = transformer_layers_count

        self.patch_count = (image_size // patch_size) ** 2
        num_classes = len(os.listdir(dataset_path))

        vit_dataset = VisionTransformerDataset(
            root_dir=dataset_path,
            image_channels=image_channels,
            image_size=image_size,
        )
        self.train_loader = DataLoader(
            dataset=vit_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self.vit_model = VisionTransformerModel(
            num_heads=self.num_heads,
            embedding_dim=self.patch_embedding_dim,
            transformer_layers_count=self.transformer_layers_count,
            device=self.device,
            num_patches=self.patch_count,
        )
        self.mlp_head = ClassificationMLP(
            num_classes=num_classes,
            embedding_dim=self.patch_embedding_dim,
            patch_count=self.patch_count,
        )

        self.vit_model.to(self.device)
        self.mlp_head.to(self.device)

        self.optim = torch.optim.Adam(params=self.vit_model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        self.writer_predictions = SummaryWriter(f"logs/predictions")
        self.step = 0

        self.nrows = 4
        self.ncols = 4

        self.start_epoch = 0
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.vit_model.load_state_dict(checkpoint['vit_model_state_dict'])
        self.mlp_head.load_state_dict(checkpoint['mlp_head_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loss = 0.0
            # correct_preds = 0
            # num_samples = 0
            for idx, (img, labels) in enumerate(self.train_loader):
                img = img.to(self.device)
                labels = labels.to(self.device)

                # (N C IMG_H IMG_W) -> (N C PATCH_COUNT_H PATCH_COUNT_W PATCH_SIZE_H PATCH_SIZE_W)
                # (64 3 72 72) -> (64 3 12 12 6 6) -> (64 12 12 6 6 3) -> (64 144 108)
                patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
                patches = patches.permute(0, 2, 3, 4, 5, 1)
                patches = patches.reshape(self.batch_size, self.patch_count, -1)  # TODO: torch.view

                vit_output = self.vit_model(patches)
                predicted_labels = self.mlp_head(vit_output)

                self.optim.zero_grad()
                loss = self.loss_fn(predicted_labels, labels)
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    if idx % self.save_every == 0:
                        torch.save({
                            'epoch': epoch,
                            'vit_model_state_dict': self.vit_model.state_dict(),
                            'mlp_head_state_dict': self.mlp_head.state_dict(),
                            'optimizer_state_dict': self.optim.state_dict(),
                        }, f'checkpoints/checkpoint_{epoch}.pt')

                        self.vit_model.eval()
                        self.mlp_head.eval()

                        # correct_preds += (torch.argmax(predicted_labels, dim=1) == labels).sum().item()
                        # num_samples += predicted_labels.shape[0]

                        train_loss += loss.item() * img.size(0)
                        train_loss /= len(self.train_loader)
                        print(
                            f"EPOCH: [{epoch} / {self.num_epochs}], BATCH: [{idx} / {len(self.train_loader)}],",
                            f"LOSS(Mean): {train_loss:.4f}",
                            # f"ACCURACY: {(100 * float(correct_preds) / num_samples):.2f}"
                        )

                        fig, ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)
                        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                        k = 0
                        for i in range(self.nrows):
                            for j in range(self.ncols):
                                ax[i, j].imshow(
                                    (img[k].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
                                )
                                ax[i, j].text(
                                    0.0, -2.0,
                                    f'GT:{labels[k]}, Prd:{torch.argmax(predicted_labels[k])}',
                                    fontsize=12
                                )
                                k += 1

                        self.writer_predictions.add_figure('Real vs Pred', figure=fig, global_step=self.step)
                        self.step += 1
                        self.vit_model.train(True)
                        self.mlp_head.train(True)


if __name__ == '__main__':
    trainer = Trainer(
        dataset_path=r'C:\staging\data',
        # checkpoint_path='checkpoints/checkpoint_18.pt',
    )
    trainer.train()
