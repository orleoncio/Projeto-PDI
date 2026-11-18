from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn


def create_vgg16(num_classes=6, pretrained=False):
    """
    Cria a arquitetura VGG-16 (Configuração D - Simonyan & Zisserman, 2014)
    seguindo o padrão do artigo de referência.

    A rede possui 16 camadas com pesos treináveis:
    - 13 camadas convolucionais com filtros 3x3 (stride=1, padding=1)
      distribuídas em 5 blocos seguidos de MaxPool2d (2x2, stride=2).
    - 3 camadas totalmente conectadas (classifier): 25088 -> 4096 -> 4096 -> num_classes.

    Parâmetros:
        num_classes (int): Número de classes de saída (padrão: 6 para o dataset HEp-2).
        pretrained (bool): Se True, carrega pesos pré-treinados no ImageNet.

    Retorno:
        nn.Module: Instância da VGG-16 ajustada para `num_classes`.
    """
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = vgg16(weights=weights)

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    return model
