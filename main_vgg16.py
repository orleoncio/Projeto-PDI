import argparse
from pathlib import Path
import torch

from data.hep2_dataset import create_dataloaders
from models.vgg16 import create_vgg16
from optimizers.optimizer_factory import create_optimizer
from training.train import train_model
from training.test import test_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação da VGG-16 no dataset HEp-2"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Otimizador a ser utilizado: 'adam' ou 'sgd' (padrão: adam)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de épocas de treinamento (padrão: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote / batch size (padrão: 32)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Apenas avalia um checkpoint já salvo em results/ sem retreinar"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    optimizer_name = args.optimizer.lower()

    # Dispositivo
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {device}")
    print(f"Arquitetura: VGG-16 | Otimizador: {optimizer_name.upper()} | Épocas: {epochs} | Batch size: {batch_size}")

    # Carregamento dos dados
    train_loader, test_loader = create_dataloaders(
        batch_size=batch_size
    )

    print(f"Imagens de treinamento: {len(train_loader.dataset)}")
    print(f"Imagens de teste:       {len(test_loader.dataset)}")

    # Criação da VGG-16
    model = create_vgg16(num_classes=6)
    model = model.to(device)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    weights_path = results_dir / f"vgg16_original_{optimizer_name}.pth"

    if not args.eval_only:
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
        torch.save(
            model.state_dict(),
            weights_path
        )
        print(f"Treinamento finalizado! Pesos salvos em: {weights_path}")
    else:
        print(f"Carregando pesos de: {weights_path}")
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device,
                weights_only=True
            )
        )

    # Avaliação no conjunto de teste (Accuracy, Precision, Recall, F1-score)
    test_model(
        model=model,
        test_loader=test_loader,
        device=device
    )


if __name__ == "__main__":
    main()

