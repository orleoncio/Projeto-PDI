import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path mesmo ao executar python tests/test_vgg16.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from models.vgg16 import create_vgg16
from optimizers.optimizer_factory import create_optimizer


def test_vgg16_architecture():
    print("--- Teste da Arquitetura VGG-16 ---")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Dispositivo: {device}")

    model = create_vgg16(num_classes=6, pretrained=False).to(device)
    print(f"Modelo: {model.__class__.__name__}")

    # 1. Verifica as 13 camadas convolucionais (3x3) e 3 camadas lineares (total = 16 camadas)
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    linear_layers = [m for m in model.classifier if isinstance(m, nn.Linear)]
    total_weight_layers = len(conv_layers) + len(linear_layers)

    print(f"Camadas convolucionais (features): {len(conv_layers)}")
    print(f"Camadas totalmente conectadas (classifier): {len(linear_layers)}")
    print(f"Total de camadas com pesos: {total_weight_layers}")

    assert len(conv_layers) == 13, f"Esperado 13 camadas Conv2d, encontrado {len(conv_layers)}"
    assert len(linear_layers) == 3, f"Esperado 3 camadas Linear, encontrado {len(linear_layers)}"
    assert total_weight_layers == 16, f"Esperado 16 camadas no total, encontrado {total_weight_layers}"
    assert all(c.kernel_size == (3, 3) for c in conv_layers), "Todas as convoluções da VGG-16 devem usar filtros 3x3"
    assert linear_layers[-1].out_features == 6, "A última camada deve produzir 6 saídas (classes HEp-2)"

    # 2. Testa integração com os otimizadores do artigo (Adam e SGD)
    opt_adam = create_optimizer("adam", model)
    opt_sgd = create_optimizer("sgd", model)
    print(f"Otimizador Adam: {opt_adam.__class__.__name__}")
    print(f"Otimizador SGD:  {opt_sgd.__class__.__name__}")

    # 3. Simula um lote com 2 imagens RGB 224x224
    images = torch.randn(2, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    print(f"Formato da saída: {outputs.shape}")
    assert outputs.shape == (2, 6), f"Formato de saída inesperado: {outputs.shape}"
    print("Sucesso: Arquitetura VGG-16 validada com êxito!")


if __name__ == "__main__":
    test_vgg16_architecture()

