import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data.hep2_dataset import Hep2Dataset
from models.vgg16 import create_vgg16
from optimizers.optimizer_factory import create_optimizer
from training.test import test_model
from preprocessing import get_preprocessing_transform


def get_augmented_transforms(preprocessing="none"):
    """
    Transformações com Data Augmentation para o treino:
    - Invariância espacial/celular: espelhamentos horizontal, vertical e rotações livres.
    - Preserva o pré-processamento original especificado.
    """
    train_ops = []
    if preprocessing != "none":
        train_ops.append(get_preprocessing_transform(preprocessing))

    train_ops.extend([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor()
    ])

    test_ops = []
    if preprocessing != "none":
        test_ops.append(get_preprocessing_transform(preprocessing))

    test_ops.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def create_augmented_dataloaders(batch_size=32, preprocessing="none", num_workers=2, pin_memory=True):
    train_transform, test_transform = get_augmented_transforms(preprocessing)

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


def train_vgg16_anti_overfitting(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    epochs=30,
    save_path="results/vgg16_optimized_sgd.pth"
):
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_test_acc = 0.0
    best_metrics = None

    print(f"\n--- Iniciando Treinamento Anti-Overfitting (30 épocas) ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        # Atualiza a taxa de aprendizado conforme o scheduler
        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        current_lr = optimizer.param_groups[0]["lr"]

        # Avaliação de validação/teste ao final de cada época
        test_results = test_model(model, test_loader, device)
        test_acc = test_results["accuracy"]

        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
            best_metrics = test_results
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)

        star = " ★ [BEST]" if is_best else ""
        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | "
            f"Test Acc: {test_acc*100:.2f}% | LR: {current_lr:.6f}{star}",
            flush=True
        )

    print(f"\n================ RESUMO DO MELHOR MODELO ================")
    print(f"Melhor Acurácia no Teste: {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision:                {best_metrics['precision']*100:.2f}%")
    print(f"Recall:                   {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:                 {best_metrics['f1']*100:.2f}%")
    print(f"Pesos salvos em:          {save_path}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Treino anti-overfitting da VGG-16 para quebrar o gap com o artigo")
    parser.add_argument("--preprocessing", type=str, default="none", choices=["none", "contrast_expansion", "histogram_equalization"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularização L2 para impedir memorização de ruído")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader, test_loader = create_augmented_dataloaders(
        batch_size=args.batch_size,
        preprocessing=args.preprocessing,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # VGG-16 treinada do zero (weights=None), respeitando a arquitetura
    model = create_vgg16(num_classes=6, pretrained=False).to(device)

    # SGD com momentum e weight decay ativo para regularização
    optimizer = create_optimizer(
        "sgd",
        model,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Cosine Annealing: reduz o learning rate gradualmente para refinamento fino
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    prep_tag = "original" if args.preprocessing == "none" else args.preprocessing
    save_path = f"results/vgg16_optimized_{prep_tag}_sgd.pth"

    train_vgg16_anti_overfitting(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
