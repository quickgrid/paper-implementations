import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from pytorch_siamese_triplet import SiameseModelTriplet, SiameseDatasetTriplet
from pytorch_siamese_triplet import calculate_distances


def infer():
    embedding_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = SiameseModelTriplet(
        embedding_dim=embedding_dim,
        device=device,
    )
    model.load_state_dict(
        torch.load('checkpoints/checkpoint_8.pt')['model_state_dict']
    )
    model.eval()
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = SiameseDatasetTriplet(
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
        anchor_image_batch = data['anchor_image']
        postive_image_batch = data['postive_image']
        negative_image_batch = data['negative_image']

        anchor_image_batch = anchor_image_batch.to(device)
        postive_image_batch = postive_image_batch.to(device)
        negative_image_batch = negative_image_batch.to(device)

        embedding_1 = model.forward_single_minibatch(anchor_image_batch)
        embedding_2 = model.forward_single_minibatch(postive_image_batch)
        embedding_3 = model.forward_single_minibatch(negative_image_batch)

        with torch.no_grad():
            ap_distance, an_distance = calculate_distances(
                anchor_embedding=embedding_1,
                positive_embedding=embedding_2,
                negative_embedding=embedding_3,
            )
            print(ap_distance, an_distance)

            cosine_similarity = nn.CosineSimilarity()
            ap_cosine_similarity = cosine_similarity(embedding_1, embedding_2)
            an_cosine_similarity = cosine_similarity(embedding_1, embedding_3)
            print(f'Similarity Score between AP: {ap_cosine_similarity}')
            print(f'Similarity Score between AP: {an_cosine_similarity}')

            fig, ax = plt.subplots(nrows, ncols * 3)
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            k = 0
            for i in range(nrows):
                m = 0
                for j in range(ncols):
                    ax[i, m].imshow(np.transpose(anchor_image_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                    ax[i, m].text(0.0, -2.0, f'anchor', fontsize=16)
                    ax[i, m + 1].imshow(np.transpose(postive_image_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                    ax[i, m + 1].text(0.0, -2.0, f'positive: {ap_cosine_similarity[k]:.2f}', fontsize=16)
                    ax[i, m + 2].imshow(np.transpose(negative_image_batch[k].detach().cpu().numpy(), (1, 2, 0)))
                    ax[i, m + 2].text(0.0, -2.0, f'negative: {an_cosine_similarity[k]:.2f}', fontsize=16)
                    m += 3
                    k += 1
            plt.show()


if __name__ == '__main__':
    infer()
