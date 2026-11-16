from pathlib import Path

import torch

from data.hep2_dataset import create_dataloaders
from models.squeezenet import create_squeezenet
from optimizers.optimizer_factory import create_optimizer
from training.train import train_model


def main():
    # Mesmas configurações utilizadas no treinamento da ResNet-50.
    epochs = 30
    batch_size = 32
    optimizer_name = "adam"  # ou "sgd"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Dispositivo: {device}")

    train_loader, _ = create_dataloaders(
        batch_size=batch_size
    )
    print(f"Imagens de treinamento: {len(train_loader.dataset)}")

    model = create_squeezenet(num_classes=6)
    model = model.to(device)

    optimizer = create_optimizer(
        optimizer_name,
        model,
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    weights_path = results_dir / f"squeezenet_original_{optimizer_name}.pth"
    torch.save(model.state_dict(), weights_path)

    print(f"Treinamento finalizado! Pesos salvos em: {weights_path}")


if __name__ == "__main__":
    main()
