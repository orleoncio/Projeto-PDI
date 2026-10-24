import torch.optim as optim
from optimizers.sgd import SGDOptimizer


def create_optimizer(
    name,
    model,
    lr=None,
    momentum=0.9,
    weight_decay=0.0,
    **kwargs
):
    """
    Fábrica para criação de otimizadores configuráveis.

    Parâmetros:
        name (str): Nome do otimizador ('adam', 'sgd', etc.).
        model (nn.Module): Modelo cujos parâmetros serão otimizados.
        lr (float, opcional): Taxa de aprendizagem. Se None, usa o padrão:
            - Adam: 0.001 (1e-3)
            - SGD: 0.01 (1e-2)
        momentum (float, opcional): Fator de momentum (padrão: 0.9 para SGD).
        weight_decay (float, opcional): Coeficiente de regularização L2 (padrão: 0.0).
        **kwargs: Argumentos extras passados ao construtor do otimizador.
    """
    name = name.lower()

    if name == "adam":
        adam_lr = lr if lr is not None else 1e-3
        return optim.Adam(
            model.parameters(),
            lr=adam_lr,
            weight_decay=weight_decay,
            **kwargs
        )

    elif name == "sgd":
        sgd_lr = lr if lr is not None else 0.01
        return SGDOptimizer(
            model.parameters(),
            lr=sgd_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )

    else:
        raise ValueError(
            f"Otimizador não suportado: {name}"
        )