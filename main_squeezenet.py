import argparse
from pathlib import Path

import torch

from data.hep2_dataset import create_dataloaders
from models.squeezenet import create_squeezenet
from optimizers.optimizer_factory import create_optimizer
from training.test import test_model
from training.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação da SqueezeNet 1.1 no dataset HEp-2"
    )
    parser.add_argument(
        "--preprocessing",
        choices=["none", "contrast_expansion", "histogram_equalization"],
        default="none",
        help="Pré-processamento das imagens (padrão: none)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Otimizador usado no treino (padrão: adam)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de épocas de treinamento (padrão: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote (padrão: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de workers para leitura das imagens (padrão: 0)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Ativa Mixed Precision (FP16) quando houver GPU CUDA",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Avalia o checkpoint existente sem treinar novamente",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    preprocessing = args.preprocessing
    optimizer_name = args.optimizer
    prep_tag = "original" if preprocessing == "none" else preprocessing
    weights_path = Path("results") / f"squeezenet_{prep_tag}_{optimizer_name}.pth"

    print(
        f"Dispositivo: {device} | Arquitetura: SqueezeNet 1.1 (do zero) | "
        f"Pré-processamento: {preprocessing} | Otimizador: {optimizer_name.upper()} | "
        f"Épocas: {args.epochs} | Batch size: {args.batch_size} | "
        f"AMP: {args.amp and use_cuda}"
    )

    train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        preprocessing=preprocessing,
        normalize=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    print(f"Imagens de treinamento: {len(train_loader.dataset)}")
    print(f"Imagens de teste:       {len(test_loader.dataset)}")

    model = create_squeezenet(num_classes=6).to(device)

    if args.eval_only:
        print(f"Carregando pesos de: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    else:
        optimizer = create_optimizer(optimizer_name, model)
        model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            use_amp=args.amp,
        )

        weights_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"Treinamento finalizado! Pesos salvos em: {weights_path}")

    test_model(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
