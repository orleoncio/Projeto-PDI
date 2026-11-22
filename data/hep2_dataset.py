import random
import pandas as pd
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocessing import get_preprocessing_transform


class RandomRotate90:
    """
    Rotaciona a imagem PIL em múltiplos exatos de 90 graus (0, 90, 180 ou 270).
    Em imagens quadradas de células HEp-2 (78x78), preserva 100% dos pixels
    sem criar bordas pretas artificiais nem distorção por interpolação.
    """

    def __call__(self, img):
        angle = random.choice((0, 90, 180, 270))
        return img if angle == 0 else img.rotate(angle)


class Hep2Dataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.data = pd.read_csv(
            csv_file,
            header=None,
            names=["filename", "label"]
        )

        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_name = str(row["filename"])
        label = int(row["label"])

        image_path = self.image_dir / image_name

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_original_transform(preprocessing="none", normalize=False):
    """Transformação determinística usada na avaliação/teste."""
    ops = []
    if preprocessing != "none":
        ops.append(get_preprocessing_transform(preprocessing))

    ops.extend([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.ToTensor()
    ])

    if normalize:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(ops)


def get_train_transform(preprocessing="none", normalize=False, augment=True):
    """
    Transformação de treinamento com Data Augmentation espacial para células HEp-2
    (invariância a orientação microscópica: rotações 90° e espelhamentos H/V),
    seguindo a metodologia de Rodrigues et al. (2020) [11] citada no artigo.
    """
    ops = []
    if preprocessing != "none":
        ops.append(get_preprocessing_transform(preprocessing))

    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90()
        ])

    ops.extend([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.ToTensor()
    ])
    if normalize:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    return transforms.Compose(ops)

def create_dataloaders(
    batch_size=32,
    preprocessing="none",
    normalize=False,
    augment=True,
    num_workers=0,
    pin_memory=False
):
    train_transform = get_train_transform(
        preprocessing=preprocessing,
        normalize=normalize,
        augment=augment
    )
    test_transform = get_original_transform(
        preprocessing=preprocessing,
        normalize=normalize
    )

    train_dataset = Hep2Dataset(
        csv_file="dataset/train.csv",
        image_dir="dataset/train",
        transform=train_transform
    )

    test_dataset = Hep2Dataset(
        csv_file="dataset/test.csv",
        image_dir="dataset/test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
