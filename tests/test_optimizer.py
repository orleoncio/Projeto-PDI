import torch
import torch.nn as nn
from optimizers import create_optimizer, SGDOptimizer


def test_sgd_optimizer():
    print("--- Teste do Otimizador SGD ---")

    # 1. Criação de um modelo simples para validação
    model = nn.Linear(4, 2, bias=False)
    initial_weights = model.weight.clone()

    # 2. Criação do otimizador via fábrica
    optimizer = create_optimizer("sgd", model, lr=0.1, momentum=0.9)
    print(f"Classe do otimizador: {optimizer.__class__.__name__}")
    print(f"Taxa de aprendizagem configurada: {optimizer.param_groups[0]['lr']}")
    print(f"Momentum configurado: {optimizer.param_groups[0]['momentum']}")

    # 3. Simula forward pass e função de perda
    inputs = torch.randn(2, 4)
    targets = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(f"Perda inicial: {loss.item():.4f}")

    # 4. Backward e atualização dos pesos
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. Verifica se os pesos foram devidamente atualizados
    updated_weights = model.weight
    diff = (updated_weights - initial_weights).abs().sum().item()
    print(f"Variação absoluta dos pesos após o step: {diff:.6f}")

    assert diff > 0, "Erro: Os pesos deveriam ter sido atualizados pelo SGD!"
    print("Sucesso: Otimizador SGD validado com êxito!")


if __name__ == "__main__":
    test_sgd_optimizer()
