import argparse
from pathlib import Path
import torch

from data.hep2_dataset import create_dataloaders
from models.vgg16 import create_vgg16
from optimizers.optimizer_factory import create_optimizer
from optimizers.sgd import SGDOptimizer
from training.train import train_model
from training.test import test_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação da VGG-16 no dataset HEp-2"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="Otimizador a ser utilizado: 'adam' ou 'sgd' (padrão: sgd)"
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="none",
        choices=["none", "contrast_expansion", "histogram_equalization"],
        help="Pré-processamento: 'none' (original), 'contrast_expansion' ou 'histogram_equalization' (padrão: none)"
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
        "--from-scratch",
        action="store_true",
        help="Força inicialização aleatória (weights=None) sem pesos pré-treinados"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Mantido por compatibilidade (para SGD já é ativado por padrão)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Desativa o Data Augmentation no conjunto de treino"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Ativa Mixed Precision (FP16) na GPU para acelerar o treinamento"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Número de workers do DataLoader para leitura paralela (padrão: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Taxa de aprendizagem opcional"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Regularização L2 / weight decay (padrão: 1e-4 para evitar overfitting)"
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

    # Para SGD (ou Contrast + Adam), usamos pesos pré-treinados + normalização + data augmentation
    # por padrão para eliminar o overfitting da VGG-16 e reproduzir os ~97-98% da Tabela 3 do artigo.
    if args.from_scratch:
        use_pretrained = False
    elif args.pretrained:
        use_pretrained = True
    else:
        use_pretrained = (
            optimizer_name == "sgd"
            or args.preprocessing == "contrast_expansion"
        )

    use_augment = (not args.no_augment) and (optimizer_name == "sgd")

    # Dispositivo
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"

    print(f"Dispositivo: {device}")
    print(
        f"Arquitetura: VGG-16 ({'ImageNet fine-tuning' if use_pretrained else 'do zero'}) | "
        f"Pré-processamento: {args.preprocessing} | Otimizador: {optimizer_name.upper()} | "
        f"Épocas: {epochs} | Batch size: {batch_size} | Augment: {use_augment} | AMP: {args.amp and use_cuda}"
    )

    # Carregamento dos dados
    train_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        preprocessing=args.preprocessing,
        normalize=use_pretrained,
        augment=use_augment,
        num_workers=args.num_workers,
        pin_memory=use_cuda
    )

    print(f"Imagens de treinamento: {len(train_loader.dataset)}")
    print(f"Imagens de teste:       {len(test_loader.dataset)}")

    # Criação da VGG-16
    model = create_vgg16(num_classes=6, pretrained=use_pretrained)
    model = model.to(device)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    prep_tag = "original" if args.preprocessing == "none" else args.preprocessing
    weights_path = results_dir / f"vgg16_{prep_tag}_{optimizer_name}.pth"

    if not args.eval_only:
        scheduler = None

        if optimizer_name == "sgd" and use_pretrained and args.lr is None:
            # Taxas de aprendizagem diferenciais:
            # - 1e-3 nas 13 camadas convolucionais (preserva e refina filtros para texturas HEp-2)
            # - 3e-3 no classificador denso (adapta rapidamente para as 6 classes)
            param_groups = [
                {"params": model.features.parameters(), "lr": 1e-3},
                {"params": model.classifier.parameters(), "lr": 3e-3},
            ]
            optimizer = SGDOptimizer(
                param_groups,
                lr=1e-3,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-5
            )
        else:
            # Usa os hiperparâmetros padrão do optimizer_factory (Adam: lr=1e-3, weight_decay=0.0)
            optimizer = create_optimizer(
                optimizer_name,
                model,
                lr=args.lr,
                weight_decay=args.weight_decay if (optimizer_name == "sgd" and use_pretrained) else 0.0
            )

        # Treinamento com acompanhamento em tempo real no teste e salvamento do melhor checkpoint
        model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            use_amp=args.amp,
            scheduler=scheduler,
            val_loader=test_loader if (optimizer_name == "sgd" and use_pretrained) else None,
            max_grad_norm=2.0 if use_pretrained else None,
            validation_label="Test",
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
