import argparse
import random
import statistics
from pathlib import Path

import torch

from data.hep2_dataset import create_dataloaders, create_kfold_dataloaders
from models.squeezenet import create_squeezenet
from optimizers.optimizer_factory import create_optimizer
from training.test import evaluate_model, test_model
from training.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação da SqueezeNet no dataset HEp-2"
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
        "--version",
        choices=["1.0", "1.1"],
        default="1.0",
        help="Versão da arquitetura (padrão 1.0, usada no artigo)",
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
        "--kfold",
        type=int,
        default=0,
        help="Executa K-fold estratificado no conjunto de treino (0 desativa; recomendado: 5)",
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


def create_training_optimizer(model, optimizer_name, learning_rate, weight_decay, epochs):
    optimizer = create_optimizer(
        optimizer_name,
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if optimizer_name == "sgd":
        # A implementação original reduz a taxa linearmente (poly, power=1).
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / epochs),
        )
    return optimizer, scheduler


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs deve ser pelo menos 1")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr deve ser maior que zero")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay não pode ser negativo")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction deve estar entre 0 e 1")
    if args.kfold == 1 or args.kfold < 0:
        raise ValueError("--kfold deve ser 0 (desativado) ou pelo menos 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    preprocessing = args.preprocessing
    optimizer_name = args.optimizer
    prep_tag = "original" if preprocessing == "none" else preprocessing
    learning_rate = args.lr if args.lr is not None else 1e-3
    split_tag = f"kf{args.kfold}" if args.kfold >= 2 else f"val{args.validation_fraction:g}"
    validation_description = (
        "K-fold" if args.kfold >= 2 else f"{args.validation_fraction:.0%}"
    )
    weights_path = Path("results") / (
        f"squeezenet{args.version.replace('.', '')}_{prep_tag}_{optimizer_name}"
        f"_lr{learning_rate:g}_wd{args.weight_decay:g}_{split_tag}"
        f"_ep{args.epochs}_bs{args.batch_size}_seed{args.seed}.pth"
    )

    print(
        f"Dispositivo: {device} | Arquitetura: SqueezeNet {args.version} (do zero) | "
        f"Pré-processamento: {preprocessing} | Otimizador: {optimizer_name.upper()} | "
        f"Épocas: {args.epochs} | Batch size: {args.batch_size} | "
        f"LR: {learning_rate:g} | Weight decay: {args.weight_decay:g} | "
        f"K-fold: {args.kfold or 'desativado'} | "
        f"Validação: {validation_description} | "
        f"AMP: {args.amp and use_cuda}"
    )

    folds = None
    validation_loader = None
    if args.kfold >= 2 and not args.eval_only:
        folds, train_loader, test_loader = create_kfold_dataloaders(
            n_splits=args.kfold,
            batch_size=args.batch_size,
            preprocessing=preprocessing,
            normalize=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            seed=args.seed,
        )
    elif args.eval_only or args.kfold >= 2:
        train_loader, test_loader = create_dataloaders(
            batch_size=args.batch_size,
            preprocessing=preprocessing,
            normalize=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            validation_fraction=0.0,
            seed=args.seed,
        )
    else:
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
    if validation_loader is not None:
        print(f"Imagens de validação:   {len(validation_loader.dataset)}")
    if folds is not None:
        print(f"Folds estratificados:   {len(folds)}")
    print(f"Imagens de teste final: {len(test_loader.dataset)}")

    if args.eval_only:
        model = create_squeezenet(num_classes=6, version=args.version).to(device)
        print(f"Carregando pesos de: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    elif folds is not None:
        fold_metrics = []
        best_epochs = []
        for fold_number, (fold_train_loader, fold_validation_loader) in enumerate(folds, start=1):
            fold_seed = args.seed + fold_number
            random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            fold_model = create_squeezenet(num_classes=6, version=args.version).to(device)
            fold_optimizer, fold_scheduler = create_training_optimizer(
                fold_model, optimizer_name, learning_rate, args.weight_decay, args.epochs
            )
            print(f"\n===== FOLD {fold_number}/{args.kfold} =====")
            fold_model, fold_info = train_model(
                model=fold_model,
                train_loader=fold_train_loader,
                optimizer=fold_optimizer,
                device=device,
                epochs=args.epochs,
                use_amp=args.amp,
                scheduler=fold_scheduler,
                val_loader=fold_validation_loader,
                return_training_info=True,
            )
            best_epochs.append(fold_info["best_epoch"])
            fold_metrics.append(
                evaluate_model(
                    fold_model,
                    fold_validation_loader,
                    device,
                    split_name=f"VALIDAÇÃO FOLD {fold_number}",
                )
            )
            del fold_model, fold_optimizer, fold_scheduler
            if use_cuda:
                torch.cuda.empty_cache()

        print(f"\n===== RESUMO {args.kfold}-FOLD (média ± desvio padrão) =====")
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            values = [metrics[metric_name] * 100 for metrics in fold_metrics]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{metric_name}: {mean:.2f}% ± {std:.2f} p.p.")

        # Refaz o modelo com todo o treino; o conjunto de teste continua independente.
        final_seed = args.seed + args.kfold + 1
        final_epochs = max(1, round(statistics.median(best_epochs)))
        random.seed(final_seed)
        torch.manual_seed(final_seed)
        model = create_squeezenet(num_classes=6, version=args.version).to(device)
        optimizer, scheduler = create_training_optimizer(
            model, optimizer_name, learning_rate, args.weight_decay, final_epochs
        )
        print(
            "\n===== TREINO FINAL COM TODO O TREINO "
            f"({final_epochs} épocas: mediana dos melhores folds) ====="
        )
        model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=final_epochs,
            use_amp=args.amp,
            scheduler=scheduler,
        )
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"Modelo final salvo em: {weights_path}")
    else:
        model = create_squeezenet(num_classes=6, version=args.version).to(device)
        optimizer, scheduler = create_training_optimizer(
            model, optimizer_name, learning_rate, args.weight_decay, args.epochs
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
