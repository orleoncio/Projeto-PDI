import torch
from pathlib import Path

from data.hep2_dataset import create_dataloaders
from models.resnet50 import create_resnet50
from optimizers.optimizer_factory import create_optimizer
from training.train import train_model


def main():

    # Configurações
    epochs = 30
    batch_size = 32
    optimizer_name = "adam"

    # Dispositivo
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {device}")

    # Carregamento dos dados
    train_loader, test_loader = create_dataloaders(
        batch_size=batch_size
    )

    print(
        f"Imagens de treinamento: "
        f"{len(train_loader.dataset)}"
    )

    # Criação da ResNet-50
    model = create_resnet50(num_classes=6)
    model = model.to(device)

    # Criação do otimizador
    optimizer = create_optimizer(
        optimizer_name,
        model
    )

    # Treinamento
    model = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )

    # Salvar os pesos
    Path("results").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        "results/resnet50_original_adam.pth"
    )

    print("Treinamento finalizado!")


if __name__ == "__main__":
    main()