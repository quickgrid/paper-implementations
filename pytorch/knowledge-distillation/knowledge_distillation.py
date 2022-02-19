"""Knowledge Distillation implementation in pytorch.

TODO
    - Remove duplicated code and simplify training for both teacher, student.
    - Do not load other models if not needed to reduce memory.
    - Add click library cli.
    - Make training mode parameter of train function.
    - Add cifar10 to test results.

References
    - https://keras.io/examples/vision/knowledge_distillation/
    - https://arxiv.org/abs/1503.02531
    - https://discuss.pytorch.org/t/tips-or-tricks-on-debugging-a-model-that-was-packaged-with-nn-sequential/17624/2
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
from torchvision import datasets
from torchvision.transforms import transforms
from torch.nn import functional
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class DistillationDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int,
    ) -> None:
        super(DistillationDataset, self).__init__()
        class_list = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        self.image_labels_files_list = list()
        for idx, class_name_folder in enumerate(class_list):
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


class LayerDebugger(nn.Module):
    """Place in between layers of nn.Sequential to get their shape.
    """
    def __init__(self):
        super(LayerDebugger, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Resnet18FeatureExtractor(nn.Module):
    def __init__(
            self,
            debug: bool = False,
    ) -> None:
        super(Resnet18FeatureExtractor, self).__init__()
        model = models.resnet18(pretrained=True)
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
            'layer3.1.relu_1': 'selected_layer',
        }
        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes=return_nodes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)['selected_layer']
        return x


class TeacherModel(nn.Module):
    def __init__(
            self,
            num_classes: int,
    ) -> None:
        super(TeacherModel, self).__init__()
        self.teacher_layers = nn.Sequential(
            Resnet18FeatureExtractor(),
            nn.Flatten(),
            nn.LazyLinear(out_features=512),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        teacher_logits = self.teacher_layers(img)
        return teacher_logits


class StudentModel(nn.Module):
    def __init__(
            self,
            img_channels: int,
            img_size: int,
            num_classes: int,
    ) -> None:
        super(StudentModel, self).__init__()

        self.student_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(in_features=((img_size // 4) ** 2) * 16, out_features=32),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=32, out_features=num_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        student_logits = self.student_layers(img)
        return student_logits


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            validation_dataset_path: str = None,
            teacher_checkpoint_path: str = None,
            student_checkpoint_path: str = None,
            device: str = None,
            training_mode: str = None,
            train_split_percentage: float = 0.7,
            label_loss_importance: float = 0.1,
            temperature: float = 10,
            num_epochs: int = 1000,
            batch_size: int = 64,
            save_every: int = 20,
            num_workers: int = 4,
            image_size: int = 96,
            learning_rate: float = 0.001,
            image_channels: int = 3,
            debug: bool = True,
    ) -> None:
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_every = save_every
        self.training_mode = training_mode
        self.label_loss_importance = label_loss_importance
        self.temperature = temperature

        class_names = os.listdir(dataset_path)
        num_classes = len(class_names)

        def _dataset_split():
            _vit_dataset = DistillationDataset(
                root_dir=dataset_path,
                image_channels=image_channels,
                image_size=image_size,
            )
            if validation_dataset_path is None:
                _train_size = int(train_split_percentage * len(_vit_dataset))
                _test_size = len(_vit_dataset) - _train_size
                _train_dataset, _validation_dataset = torch.utils.data.random_split(
                    _vit_dataset,
                    [_train_size, _test_size]
                )
                return _train_dataset, _validation_dataset
            else:
                _validation_dataset = DistillationDataset(
                    root_dir=validation_dataset_path,
                    image_channels=image_channels,
                    image_size=image_size,
                )
                return _vit_dataset, _validation_dataset

        train_dataset, validation_dataset = _dataset_split()

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.teacher_model = TeacherModel(
            num_classes=num_classes,
        )
        self.student_model = StudentModel(
            img_channels=image_channels,
            img_size=image_size,
            num_classes=num_classes,
        )

        if debug:
            m = torch.rand(1, image_channels, image_size, image_size)
            self.teacher_model(m)
            self.student_model(m)
            num_params_student = sum(param.numel() for param in self.student_model.parameters() if param.requires_grad)
            num_params_teacher = sum(param.numel() for param in self.teacher_model.parameters() if param.requires_grad)
            print(f'STUDENT PARAMS: {num_params_student}, TEACHER PARAMS: {num_params_teacher}')

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        self.optim = torch.optim.Adam(
            params=self.teacher_model.parameters(), lr=learning_rate, weight_decay=0.001
        )

        self.label_crossentropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.soft_target_distillation_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        # self.soft_target_distillation_loss = nn.MSELoss().to(self.device)

        self.writer_predictions = SummaryWriter(f"logs/predictions")
        self.step = 0

        self.nrows = 4
        self.ncols = 4

        assert training_mode in ['teacher', 'student', 'distill'], "Invalid training mode."

        self.teacher_start_epoch = 0
        self.student_start_epoch = 0
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True)
        if teacher_checkpoint_path is not None:
            self.load_teacher_checkpoint(checkpoint_path=teacher_checkpoint_path)
        if training_mode == 'teacher':
            self.start_epoch = self.teacher_start_epoch
            self.selected_model = self.teacher_model
        if training_mode == 'distill' or training_mode == 'student':
            if student_checkpoint_path is not None:
                self.load_student_checkpoint(checkpoint_path=student_checkpoint_path)
                self.start_epoch = self.student_start_epoch
            else:
                self.start_epoch = 0
            self.selected_model = self.student_model

    def load_teacher_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.teacher_start_epoch = checkpoint['epoch']

    def load_student_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.student_start_epoch = checkpoint['epoch']

    def train(self) -> None:
        if self.training_mode == 'distill':
            self.distill()
        else:
            self.train_individual()

    def distill(self) -> None:
        self.teacher_model.eval()

        best_accuracy = 0.0
        for epoch in range(self.start_epoch, self.num_epochs):
            # Training loop.
            running_student_label_loss = 0.0
            running_student_distill_loss = 0.0
            training_correct_preds = 0
            training_total_data = 0
            with tqdm(self.train_loader) as tqdm_train_loader:
                tqdm_train_loader.set_description(f'TRAIN EPOCH: {epoch} ')
                for idx, (img, labels) in enumerate(tqdm_train_loader):
                    img = img.to(self.device)
                    labels = labels.to(self.device)

                    student_logits = self.student_model(img)
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(img)

                    self.optim.zero_grad()
                    student_label_loss = self.label_crossentropy_loss(student_logits, labels)
                    distillation_loss = self.soft_target_distillation_loss(
                        functional.log_softmax(teacher_logits / self.temperature, dim=1),
                        functional.softmax(student_logits / self.temperature, dim=1),
                    )
                    loss = self.label_loss_importance * student_label_loss + (
                            1 - self.label_loss_importance) * distillation_loss
                    loss.backward()
                    self.optim.step()

                    running_student_label_loss += student_label_loss.item()
                    running_student_distill_loss += distillation_loss.item()
                    training_total_data += labels.shape[0]
                    training_correct_preds += (torch.argmax(student_logits, dim=1) == labels).sum().item()

            # Validation loop.
            running_validation_label_loss = 0.0
            running_validation_distill_loss = 0.0
            validation_correct_preds = 0
            validation_total_data = 0
            with torch.no_grad():
                self.selected_model.eval()
                with tqdm(self.validation_loader) as tqdm_validation_loader:
                    tqdm_validation_loader.set_description(f'VALID EPOCH: {epoch} ')
                    for idx, (img, labels) in enumerate(tqdm_validation_loader):
                        img = img.to(self.device)
                        labels = labels.to(self.device)

                        student_logits = self.student_model(img)
                        teacher_logits = self.teacher_model(img)

                        student_label_loss = self.label_crossentropy_loss(student_logits, labels)
                        distillation_loss = self.soft_target_distillation_loss(
                            functional.log_softmax(teacher_logits / self.temperature, dim=1),
                            functional.softmax(student_logits / self.temperature, dim=1),
                        )
                        # validation_loss = self.label_loss_importance * student_label_loss + (
                        #         1 - self.label_loss_importance) * distillation_loss

                        running_validation_label_loss += student_label_loss.item()
                        running_validation_distill_loss += distillation_loss.item()
                        validation_total_data += img.shape[0]
                        validation_correct_preds += (torch.argmax(student_logits, dim=1) == labels).sum().item()

                        if idx % self.save_every == self.save_every - 1:
                            fig, ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)
                            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                            k = 0
                            for i in range(self.nrows):
                                for j in range(self.ncols):
                                    ax[i, j].imshow(
                                        (img[k].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(
                                            torch.uint8).detach().cpu().numpy()
                                    )
                                    ax[i, j].text(
                                        0.0, -2.0,
                                        f'GT:{labels[k]}, Prd:{torch.argmax(student_logits[k])}',
                                        fontsize=12
                                    )
                                    k += 1

                            self.writer_predictions.add_figure(
                                'Validation Real vs Pred', figure=fig, global_step=self.step
                            )
                            self.step += 1

                    self.selected_model.train()

                train_label_loss = running_student_label_loss / len(self.train_loader)
                train_distill_loss = running_student_distill_loss / len(self.train_loader)
                training_accuracy = (100.0 * training_correct_preds / training_total_data)
                validation_label_loss_value = running_validation_label_loss / len(self.validation_loader)
                validation_distill_loss_value = running_validation_distill_loss / len(self.validation_loader)
                validation_accuracy = (100.0 * validation_correct_preds / validation_total_data)

                print(
                    f"TRAIN -> [LABEL, DISTILL] "
                    f"LOSS: [{train_label_loss:.3f}, {train_distill_loss:.3f}], "
                    f"ACCURACY: {training_accuracy:.3f}, "
                    f"VALID -> [LABEL, DISTILL] "
                    f"LOSS: [{validation_label_loss_value:.3f}, {validation_distill_loss_value:.3f}] "
                    f"ACCURACY: {validation_accuracy:.3f}"
                )

                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.selected_model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                    }, f'checkpoints/{self.training_mode}_checkpoint_{epoch}.pt')

    def train_individual(self) -> None:
        best_accuracy = 0.0
        for epoch in range(self.start_epoch, self.num_epochs):
            # Training loop.
            running_train_loss = 0.0
            training_correct_preds = 0
            training_total_data = 0
            with tqdm(self.train_loader) as tqdm_train_loader:
                tqdm_train_loader.set_description(f'TRAIN EPOCH: {epoch} ')
                for idx, (img, labels) in enumerate(tqdm_train_loader):
                    img = img.to(self.device)
                    labels = labels.to(self.device)

                    predicted_labels = self.selected_model(img)

                    self.optim.zero_grad()
                    loss = self.label_crossentropy_loss(predicted_labels, labels)
                    loss.backward()
                    self.optim.step()

                    running_train_loss += loss.item()
                    training_total_data += labels.shape[0]
                    training_correct_preds += (torch.argmax(predicted_labels, dim=1) == labels).sum().item()

            # Validation loop.
            running_validation_loss = 0.0
            validation_correct_preds = 0
            validation_total_data = 0
            with torch.no_grad():
                self.selected_model.eval()
                with tqdm(self.validation_loader) as tqdm_validation_loader:
                    tqdm_validation_loader.set_description(f'VALID EPOCH: {epoch} ')
                    for idx, (img, labels) in enumerate(tqdm_validation_loader):
                        img = img.to(self.device)
                        labels = labels.to(self.device)

                        predicted_labels = self.selected_model(img)
                        validation_loss = self.label_crossentropy_loss(predicted_labels, labels)

                        running_validation_loss += validation_loss.item()
                        validation_total_data += img.shape[0]
                        validation_correct_preds += (torch.argmax(predicted_labels, dim=1) == labels).sum().item()

                        if idx % self.save_every == self.save_every - 1:
                            fig, ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)
                            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                            k = 0
                            for i in range(self.nrows):
                                for j in range(self.ncols):
                                    ax[i, j].imshow(
                                        (img[k].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(
                                            torch.uint8).detach().cpu().numpy()
                                    )
                                    ax[i, j].text(
                                        0.0, -2.0,
                                        f'GT:{labels[k]}, Prd:{torch.argmax(predicted_labels[k])}',
                                        fontsize=12
                                    )
                                    k += 1

                            self.writer_predictions.add_figure(
                                'Validation Real vs Pred', figure=fig, global_step=self.step
                            )
                            self.step += 1

                    self.selected_model.train()

                train_loss_value = running_train_loss / len(self.train_loader)
                training_accuracy = (100.0 * training_correct_preds / training_total_data)
                validation_loss_value = running_validation_loss / len(self.validation_loader)
                validation_accuracy = (100.0 * validation_correct_preds / validation_total_data)

                print(
                    f"TRAINING LOSS: {train_loss_value:.3f}, "
                    f"TRAINING ACCURACY: {training_accuracy:.3f}, "
                    f"VALIDATION LOSS: {validation_loss_value:.3f}, "
                    f"VALIDATION ACCURACY: {validation_accuracy:.3f}"
                )

                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.selected_model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                    }, f'checkpoints/{self.training_mode}_checkpoint_{epoch}.pt')


if __name__ == '__main__':
    trainer = Trainer(
        dataset_path=r'C:\staging\classification_data',
        # teacher_checkpoint_path='checkpoints/teacher_checkpoint_5.pt',
        # student_checkpoint_path='checkpoints/distill_checkpoint_7.pt',
        image_size=96,
        batch_size=32,
        # training_mode='distill',
        # training_mode='student',
        training_mode='teacher',
    )
    trainer.train()
