"""Siamese Network implementation with triplet loss.

Two or more identical networks that produces similar shape outputs/embeddings.
These embeddings are used to idenify if two things are same or similar.

Notes
    - Add LazyLinear alternative by getting feature dimension from dummy data pass.

References
    - https://github.com/fangpin/siamese-pytorch
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - https://discuss.pytorch.org/t/dataloader-for-a-siamese-model-with-concatdataset/66085
    - https://keras.io/examples/vision/siamese_network/
    - https://keras.io/examples/vision/siamese_contrastive/
    - https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
    - https://pytorch.org/vision/master/feature_extraction.html
    - https://github.com/quickgrid/code-lab/blob/master/code-lab/pytorch/pytorch_siamese_network.py
"""
import os
from typing import Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor


class Resnet18FeatureExtractor(nn.Module):
    def __init__(
            self,
            device
    ):
        super(Resnet18FeatureExtractor, self).__init__()
        model = models.resnet18(pretrained=True).to(device=device)
        model.train(mode=False)

        # train_nodes, eval_nodes = get_graph_node_names(model)
        # print('train_nodes')
        # print(train_nodes)
        # print('eval_nodes')
        # print(eval_nodes)

        return_nodes = {
            'layer3.1.relu_1': 'layer3',
        }
        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes=return_nodes
        ).to(device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SiameseModelTriplet(nn.Module):
    def __init__(
            self,
            device: torch.device,
            embedding_dim: int = 128,
    ) -> None:
        super(SiameseModelTriplet, self).__init__()
        self.base_network = Resnet18FeatureExtractor(device=device)
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=embedding_dim),
        )

    def forward_single_minibatch(self, x: Tensor) -> Tensor:
        x = self.base_network(x)
        x = self.flatten(x['layer3'])
        x = self.output(x)
        return x

    def forward(self, achor: Tensor, positive: Tensor, negative: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        achor_embedding = self.forward_single_minibatch(achor)
        positive_embedding = self.forward_single_minibatch(positive)
        negative_embedding = self.forward_single_minibatch(negative)
        return (
            achor_embedding,
            positive_embedding,
            negative_embedding
        )


class SiameseDatasetTriplet(Dataset):
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

        # Here, I have made pairings based on total file count. This pairing can be changed to N.
        self.triplet_file_paths = list()
        for i in range(total_files):
            # Anchor and positive images come from same folder but both are not same files.
            anchor_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            selected_anchor_folder = index_to_class_folder_map[anchor_folder_idx]
            anchor_image_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_anchor_folder]
            )
            positive_image_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_anchor_folder]
            )
            while anchor_image_index == positive_image_index:
                positive_image_index = np.random.randint(
                    low=0,
                    high=folder_name_file_count_dict[selected_anchor_folder]
                )

            # Negative image selection such that it does not belong to anchor, positive class.
            negative_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            while negative_folder_idx == anchor_folder_idx:
                negative_folder_idx = np.random.randint(low=0, high=self.class_names_len)
            selected_negative_folder = index_to_class_folder_map[negative_folder_idx]
            negative_image_index = np.random.randint(
                low=0,
                high=folder_name_file_count_dict[selected_negative_folder]
            )

            # print(
            #     name_label_dict.get(selected_anchor_folder)[anchor_image_index],
            #     name_label_dict.get(selected_anchor_folder)[positive_image_index],
            #     name_label_dict.get(selected_negative_folder)[negative_image_index]
            # )

            # Insert (anchor, positive, negative) triplet into dictionary.
            self.triplet_file_paths.append(
                (
                    os.path.join(self.root_dir, selected_anchor_folder,
                                 name_label_dict.get(selected_anchor_folder)[anchor_image_index]),
                    os.path.join(self.root_dir, selected_anchor_folder,
                                 name_label_dict.get(selected_anchor_folder)[positive_image_index]),
                    os.path.join(self.root_dir, selected_negative_folder,
                                 name_label_dict.get(selected_negative_folder)[negative_image_index]),
                )
            )

        self.triplet_file_names_len = len(self.triplet_file_paths)

    def __len__(self) -> int:
        return self.triplet_file_names_len

    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, Union[Image.Image, int, bool]]:
        anchor_image_path, positive_image_path, negative_image_path = self.triplet_file_paths[idx]

        anchor_image = Image.open(anchor_image_path)
        postive_image = Image.open(positive_image_path)
        negative_image = Image.open(negative_image_path)
        anchor_image = anchor_image.convert('RGB')
        postive_image = postive_image.convert('RGB')
        negative_image = negative_image.convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            postive_image = self.transform(postive_image)
            negative_image = self.transform(negative_image)

        sample = {
            'anchor_image': anchor_image,
            'postive_image': postive_image,
            'negative_image': negative_image,
        }
        return sample


def calculate_distances(
        anchor_embedding: Tensor,
        positive_embedding: Tensor,
        negative_embedding: Tensor
) -> Tuple[Tensor, Tensor]:
    ap_distance = torch.sum(torch.square(anchor_embedding - positive_embedding), dim=-1)
    an_distance = torch.sum(torch.square(anchor_embedding - negative_embedding), dim=-1)
    return ap_distance, an_distance


def triplet_loss_fn(margin: float):
    def triplet_loss(ap_distance: Tensor, an_distance: Tensor) -> Tensor:
        return torch.mean(
            torch.maximum(ap_distance - an_distance + margin, torch.zeros_like(ap_distance))
        )

    return triplet_loss


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            batch_size: int = 32,
            num_workers: int = 8,
            learning_rate: float = 0.0001,
            num_epochs: int = 1000,
            embedding_dim: int = 128,
    ):
        super(Trainer, self).__init__()
        self.num_epochs = num_epochs

        data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = SiameseDatasetTriplet(
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
        self.model = SiameseModelTriplet(embedding_dim=embedding_dim, device=self.device).to(self.device)
        self.loss_fn = triplet_loss_fn(margin=0.5)
        self.optimizer_fn = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            for batch_idx, data in enumerate(self.train_loader):
                anchor_image_batch = data['anchor_image']
                postive_image_batch = data['postive_image']
                negative_image_batch = data['negative_image']

                anchor_image_batch = anchor_image_batch.to(self.device)
                postive_image_batch = postive_image_batch.to(self.device)
                negative_image_batch = negative_image_batch.to(self.device)

                anchor_embedding, positive_embedding, negative_embedding = self.model.forward(
                    anchor_image_batch,
                    postive_image_batch,
                    negative_image_batch,
                )
                ap_distance, an_distance = calculate_distances(
                    anchor_embedding,
                    positive_embedding,
                    negative_embedding
                )

                loss = self.loss_fn(ap_distance, an_distance)
                self.optimizer_fn.zero_grad()
                loss.backward()
                self.optimizer_fn.step()

                train_loss += loss.item() * anchor_image_batch.size(0)

                # fig, ax = plt.subplots(1, 3)
                # ax[0].imshow(np.transpose(anchor_image_batch[0].detach().cpu().numpy(), (1, 2, 0)))
                # ax[1].imshow(np.transpose(postive_image_batch[0].detach().cpu().numpy(), (1, 2, 0)))
                # ax[2].imshow(np.transpose(negative_image_batch[0].detach().cpu().numpy(), (1, 2, 0)))
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
        dataset_path='../dataset/mnist_sample'
    )
    trainer.train()
