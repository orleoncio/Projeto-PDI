import argparse
import random
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
        "--lr",
        type=float,
        default=None,
        help="Learning rate (padrão: 1e-3 para Adam e SGD)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-4,
        help="Penalização L2 dos pesos (padrão inspirado na implementação SqueezeNet)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fração estratificada do treino reservada para validação (padrão: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente da divisão de validação e das transformações aleatórias",
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
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Pula a avaliação no conjunto de teste durante comparação de hiperparâmetros",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr deve ser maior que zero")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay não pode ser negativo")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction deve estar entre 0 e 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    preprocessing = args.preprocessing
    optimizer_name = args.optimizer
    prep_tag = "original" if preprocessing == "none" else preprocessing
    learning_rate = args.lr if args.lr is not None else 1e-3
    weights_path = Path("results") / (
        f"squeezenet_{prep_tag}_{optimizer_name}_lr{learning_rate:g}"
        f"_wd{args.weight_decay:g}_val{args.validation_fraction:g}"
        f"_seed{args.seed}.pth"
    )

    print(
        f"Dispositivo: {device} | Arquitetura: SqueezeNet 1.1 (do zero) | "
        f"Pré-processamento: {preprocessing} | Otimizador: {optimizer_name.upper()} | "
        f"Épocas: {args.epochs} | Batch size: {args.batch_size} | "
        f"LR: {learning_rate:g} | Weight decay: {args.weight_decay:g} | "
        f"Validação: {args.validation_fraction:.0%} | "
        f"AMP: {args.amp and use_cuda}"
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        preprocessing=preprocessing,
        normalize=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(f"Imagens de treinamento: {len(train_loader.dataset)}")
    print(f"Imagens de validação:   {len(validation_loader.dataset)}")
    print(f"Imagens de teste:       {len(test_loader.dataset)}")

    model = create_squeezenet(num_classes=6).to(device)

    if args.eval_only:
        print(f"Carregando pesos de: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    else:
        optimizer = create_optimizer(
            optimizer_name,
            model,
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = None
        if optimizer_name == "sgd":
            # A implementação original reduz a taxa linearmente (poly, power=1).
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / args.epochs),
            )

        model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            use_amp=args.amp,
            scheduler=scheduler,
            val_loader=validation_loader,
        )

        weights_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"Treinamento finalizado! Pesos salvos em: {weights_path}")

    if not args.skip_test:
        test_model(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
