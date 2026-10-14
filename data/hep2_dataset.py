import pandas as pd
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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

def get_original_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def create_dataloaders(batch_size=32):
    transform = get_original_transform()

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
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader