from torchvision.models import vgg16
import torch.nn as nn


def create_vgg16(num_classes=6):
    """
    Cria a arquitetura VGG-16 (Configuração D - Simonyan & Zisserman, 2014)
    seguindo o padrão do artigo de referência.

    A rede possui 16 camadas com pesos treináveis:
    - 13 camadas convolucionais com filtros 3x3 (stride=1, padding=1)
      distribuídas em 5 blocos seguidos de MaxPool2d (2x2, stride=2).
    - 3 camadas totalmente conectadas (classifier): 25088 -> 4096 -> 4096 -> num_classes.

    Parâmetros:
        num_classes (int): Número de classes de saída (padrão: 6 para o dataset HEp-2).

    Retorno:
        nn.Module: Instância da VGG-16 sem pesos pré-treinados ajustada para `num_classes`.
    """
    model = vgg16(weights=None)

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    return model

