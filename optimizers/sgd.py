import torch
from torch.optim import Optimizer


class SGDOptimizer(Optimizer):
    """
    Implementação do otimizador SGD (Stochastic Gradient Descent).

    O SGD atualiza os pesos do modelo iterativamente calculando o gradiente da
    função de perda em relação aos parâmetros:
        theta_{t+1} = theta_t - lr * g_t

    Características e extensões implementadas:
    - Taxa de aprendizagem (lr): controla a magnitude de cada passo.
    - Momentum: ameniza a sensibilidade à taxa de aprendizagem e acelera a
      convergência ao acumular gradientes de iterações passadas:
        v_{t+1} = momentum * v_t + g_t
        theta_{t+1} = theta_t - lr * v_{t+1}
    - Weight Decay (regularização L2): penaliza pesos de grande magnitude:
        g_t = g_t + weight_decay * theta_t
    - Nesterov Momentum: calcula o gradiente antecipando a direção do momentum.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False
    ):
        if lr < 0.0:
            raise ValueError(f"Taxa de aprendizagem inválida: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Valor de momentum inválido: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Valor de weight_decay inválido: {weight_decay}")
        if nesterov and (momentum <= 0.0):
            raise ValueError("Nesterov momentum requer momentum > 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Executa um único passo de otimização (atualização dos pesos).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # Aplica Weight Decay (regularização L2)
                if weight_decay != 0.0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Aplica Momentum
                if momentum != 0.0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Atualização dos pesos: theta = theta - lr * gradiente
                p.add_(d_p, alpha=-lr)

        return loss


# Alias para facilitar o uso
SGD = SGDOptimizer
