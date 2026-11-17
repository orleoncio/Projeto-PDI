import pandas as pd
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocessing import get_preprocessing_transform


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

    ops = []
    if preprocessing != "none":
        ops.append(get_preprocessing_transform(preprocessing))

    ops.extend([
        transforms.Resize((224, 224)),
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
    num_workers=0,
    pin_memory=False
):
    transform = get_original_transform(
        preprocessing=preprocessing,
        normalize=normalize
    )

    train_dataset = Hep2Dataset(
        csv_file="dataset/train.csv",
        image_dir="dataset/train",
        transform=transform
    )

    test_dataset = Hep2Dataset(
        csv_file="dataset/test.csv",
        image_dir="dataset/test",
        transform=transform
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
