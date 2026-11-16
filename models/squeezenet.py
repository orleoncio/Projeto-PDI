from torchvision.models import squeezenet1_1


def create_squeezenet(num_classes=6):
    """Cria uma SqueezeNet 1.1 sem pesos pré-treinados."""
    return squeezenet1_1(
        weights=None,
        num_classes=num_classes,
    )
