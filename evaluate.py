import torch

from data.hep2_dataset import create_dataloaders
from models.resnet50 import create_resnet50
from training.test import test_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {device}")

    # Carrega as imagens
    _, test_loader = create_dataloaders(
        batch_size=32
    )

    # Cria a arquitetura ResNet-50
    model = create_resnet50(num_classes=6)

    # Carrega os pesos do treinamento
    model.load_state_dict(
        torch.load(
            "results/resnet50_original_adam.pth",
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)

    # Executa o teste
    test_model(
        model=model,
        test_loader=test_loader,
        device=device
    )


if __name__ == "__main__":
    main()