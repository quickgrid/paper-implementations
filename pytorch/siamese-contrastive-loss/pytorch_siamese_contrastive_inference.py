import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from pytorch_siamese_contrastive import SiameseModelContrastive, SiameseDatasetContrastive


def infer():
    embedding_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = SiameseModelContrastive(
        embedding_dim=embedding_dim,
        device=device,
    )
    model.load_state_dict(
        torch.load('checkpoints/checkpoint_30.pt')['model_state_dict']
    )
    model.eval()
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = SiameseDatasetContrastive(
        root_dir='../dataset/mnist_sample',
        transform=data_transforms
    )
    batch_size: int = 4
    num_workers: int = 8
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Matplotlib config.
    nrows = 2
    ncols = 2

    for batch_idx, data in enumerate(train_loader):
        image1_batch = data['image1']
        image2_batch = data['image2']

        image1_batch = image1_batch.to(device)
        image2_batch = image2_batch.to(device)

        with torch.no_grad():
            similarity_values = model.forward(
                image1_batch,
                image2_batch,
            )

            print(f'Similarity Score: {similarity_values}')

            fig, ax = plt.subplots(nrows, ncols * 2)
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            k = 0
            for i in range(nrows):
                m = 0
                for j in range(ncols):
                    ax[i, m].imshow(np.transpose(image1_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                    ax[i, m].text(0.0, -2.0, f'image1', fontsize=16)
                    ax[i, m + 1].imshow(np.transpose(image2_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                    ax[i, m + 1].text(0.0, -2.0, f'image2: {similarity_values[k]:.2f}', fontsize=16)
                    m += 2
                    k += 1
            plt.show()


if __name__ == '__main__':
    infer()
