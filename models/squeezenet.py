from torchvision.models import squeezenet1_0, squeezenet1_1


def create_squeezenet(num_classes=6, version="1.0"):
    """Cria SqueezeNet 1.0 (referência do artigo) ou 1.1 sem pré-treino."""
    builders = {
        "1.0": squeezenet1_0,
        "1.1": squeezenet1_1,
    }
    try:
        builder = builders[version]
    except KeyError as exc:
        raise ValueError(f"Versão SqueezeNet inválida: {version}. Use '1.0' ou '1.1'.") from exc

    return builder(
        weights=None,
        num_classes=num_classes,
    )
