"""Siamese Network implementation with contrastive loss.

Two or more identical networks that produces similar shape outputs/embeddings.
These embeddings are used to idenify if two things are same or similar.

Notes
    - Add LazyLinear alternative by getting feature dimension from dummy data pass.
    - Make data loader return data such that there is equal distribution of similar and disimilar data.
    - Add tqdm progress bar.
    - Make embeddings with tanh activation, and L2 distance output sigmoid.

References
    - https://github.com/fangpin/siamese-pytorch
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - https://discuss.pytorch.org/t/dataloader-for-a-siamese-model-with-concatdataset/66085
    - https://keras.io/examples/vision/siamese_network/
    - https://keras.io/examples/vision/siamese_contrastive/
    - https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
    - https://pytorch.org/vision/master/feature_extraction.html
    - https://github.com/quickgrid/code-lab/blob/master/code-lab/pytorch/pytorch_siamese_network.py
    - https://github.com/quickgrid/paper-implementations/tree/main/pytorch/siamese-triplet-loss
"""
import os
from typing import Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class Resnet18FeatureExtractor(nn.Module):
    def __init__(
            self,
            device: torch.device,
            debug: bool = False,
    ):
        super(Resnet18FeatureExtractor, self).__init__()
        model = models.resnet18(pretrained=True).to(device=device)
        # model.train(mode=False)

        # for param in model.parameters():
        #     param.requires_grad = False

        if debug:
            train_nodes, eval_nodes = get_graph_node_names(model)
            print('train_nodes')
            print(train_nodes)
            print('eval_nodes')
            print(eval_nodes)

        return_nodes = {
            'layer2.1.relu_1': 'selected_layer',
        }
        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes=return_nodes
        ).to(device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SiameseModelContrastive(nn.Module):
    def __init__(
            self,
            device: torch.device,
            embedding_dim: int = 128,
    ) -> None:
        super(SiameseModelContrastive, self).__init__()
        self.base_network = Resnet18FeatureExtractor(device=device)
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=1024, out_features=embedding_dim),
        )
        self.pairwise_euclidean = nn.PairwiseDistance()
        self.distance_output = nn.Sequential(
            nn.Tanh(),
        )

    def forward_single_minibatch(self, x: Tensor) -> Tensor:
        x = self.base_network(x)
        x = self.flatten(x['selected_layer'])
        x = self.output(x)
        return x

    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        img1_embedding = self.forward_single_minibatch(img1)
        img2_embedding = self.forward_single_minibatch(img2)
        euclidean_distance = self.pairwise_euclidean(img1_embedding, img2_embedding)
        return self.distance_output(euclidean_distance)


class SiameseDatasetContrastive(Dataset):
    def __init__(
            self,
            root_dir: str,
            transform: transforms = None
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform

        class_names_folder = os.listdir(root_dir)
        self.class_names_len = len(class_names_folder)

        folder_name_file_count_dict = dict()
        index_to_class_folder_map = dict()
        name_label_dict = dict()
        total_files = 0
        for idx, class_folder in enumerate(class_names_folder):
            image_files = os.listdir(os.path.join(root_dir, class_folder))
            file_count = len(image_files)
            total_files += file_count
            folder_name_file_count_dict[class_folder] = file_count
            index_to_class_folder_map[idx] = class_folder
            name_label_dict[class_folder] = image_files

        # print(name_label_dict)
        # print(total_files)

        # Make double pairing like (anchor, positive, 1) and (anchor, negative, 0) to make concept easier.
        self.image_pair_and_label = list()
        pairings_amount = total_files
        for i in range(pairings_amount):
            anchor_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            selected_anchor_folder = index_to_class_folder_map[anchor_folder_idx]
            anchor_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_anchor_folder]
            )
            positive_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_anchor_folder]
            )
            self.image_pair_and_label.append(
                (
                    os.path.join(self.root_dir, selected_anchor_folder,
                                 name_label_dict.get(selected_anchor_folder)[anchor_index]),
                    os.path.join(self.root_dir, selected_anchor_folder,
                                 name_label_dict.get(selected_anchor_folder)[positive_index]),
                    1,
                )
            )

            # Make (anchor, negative, 0) reusing anchor image.
            negative_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            while negative_folder_idx == anchor_folder_idx:
                negative_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            selected_negative_folder = index_to_class_folder_map[negative_folder_idx]
            negative_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_negative_folder]
            )
            self.image_pair_and_label.append(
                (
                    os.path.join(self.root_dir, selected_anchor_folder,
                                 name_label_dict.get(selected_anchor_folder)[anchor_index]),
                    os.path.join(self.root_dir, selected_negative_folder,
                                 name_label_dict.get(selected_negative_folder)[negative_index]),
                    0,
                )
            )

        # print(self.image_pair_and_label)
        self.contrastive_file_names_len = len(self.image_pair_and_label)

    def __len__(self) -> int:
        return self.contrastive_file_names_len

    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, Union[Image.Image, int, bool]]:
        image1_path, image2_path, pair_label = self.image_pair_and_label[idx]

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {
            'image1': image1,
            'image2': image2,
            'pair_label': pair_label,
        }
        return sample


def contrastive_loss_fn(
        device: torch.device,
        margin: float = 1.0
) -> torch.float:
    def contrastive_loss(y_true, y_pred) -> Tensor:
        square_pred = torch.square(y_pred)
        margin_square = torch.square(torch.max(margin - y_pred, torch.zeros_like(y_pred).to(device)))
        return torch.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            checkpoint_path: str = None,
            batch_size: int = 64,
            num_workers: int = 8,
            learning_rate: float = 0.0001,
            num_epochs: int = 1000,
            embedding_dim: int = 128,
    ):
        super(Trainer, self).__init__()
        self.nrows = 2
        self.ncols = 2
        self.num_epochs = num_epochs

        data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = SiameseDatasetContrastive(
            root_dir=dataset_path,
            transform=data_transforms
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.num_classes = len(os.listdir(dataset_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseModelContrastive(embedding_dim=embedding_dim, device=self.device).to(self.device)
        self.loss_fn = contrastive_loss_fn(device=self.device, margin=1.0)
        self.optimizer_fn = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.start_epoch = 0
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_fn.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loss = 0.0
            for batch_idx, data in enumerate(self.train_loader):
                image1_batch = data['image1']
                image2_batch = data['image2']
                pair_label_batch = data['pair_label']
                # print(pair_label_batch)

                image1_batch = image1_batch.to(self.device)
                image2_batch = image2_batch.to(self.device)
                pair_label_batch = pair_label_batch.to(self.device)

                similarity_values = self.model.forward(
                    image1_batch,
                    image2_batch,
                )

                loss = self.loss_fn(pair_label_batch, similarity_values)
                self.optimizer_fn.zero_grad()
                loss.backward()
                self.optimizer_fn.step()

                train_loss += loss.item() * image1_batch.size(0)

                # fig, ax = plt.subplots(self.nrows, self.ncols * 2)
                # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                # k = 0
                # for i in range(self.nrows):
                #     m = 0
                #     for j in range(self.ncols):
                #         ax[i, m].imshow(np.transpose(image1_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                #         ax[i, m].text(0.0, -2.0, f'anchor', fontsize=16)
                #         ax[i, m + 1].imshow(np.transpose(image2_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                #         ax[i, m + 1].text(0.0, -2.0, f'Similarity {pair_label_batch[k]}', fontsize=16)
                #         m += 2
                #         k += 1
                # plt.show()

            train_loss /= len(self.train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch,
                train_loss
            ))

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer_fn.state_dict(),
                }, f'checkpoints/checkpoint_{epoch}.pt')


if __name__ == '__main__':
    # Fire(Trainer)
    trainer = Trainer(
        dataset_path='../dataset/mnist_sample',
        # checkpoint_path='checkpoints/checkpoint_10.pt',
    )
    trainer.train()
