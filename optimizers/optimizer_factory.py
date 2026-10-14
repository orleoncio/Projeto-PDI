import torch.optim as optim

def create_optimizer(name, model):

    if name == "adam":
        return optim.Adam(
            model.parameters()
        )

    elif name == "sgd":
        return optim.SGD(
            model.parameters()
        )

    else:
        raise ValueError(
            f"Otimizador não suportado: {name}"
        )