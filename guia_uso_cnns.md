# Guia de uso dos componentes compartilhados — CNNs

Este guia mostra como utilizar os arquivos já criados para treinar **outras CNNs** com o mesmo dataset e os mesmos otimizadores.

## 1. Arquivos que devem ser reutilizados

```text
data/hep2_dataset.py
optimizers/optimizer_factory.py
training/train.py
```

Cada colega cria apenas o arquivo de sua arquitetura em `models/` e adapta seu `main.py`.

## 2. Criar e importar sua CNN

A arquitetura deve ser um `torch.nn.Module` que receba imagens RGB e produza **6 logits por imagem**. Exporte uma função de criação do modelo, por exemplo:

```python
from models.minha_cnn import create_model
model = create_model(num_classes=6)
```

Substitua `minha_cnn` pelo nome do arquivo e `create_model` pela função definida nele. Não é necessário alterar os componentes compartilhados.

## 3. Carregar os dados e selecionar o otimizador

```python
import torch
from data.hep2_dataset import create_dataloaders
from optimizers.optimizer_factory import create_optimizer
from training.train import train_model
from models.minha_cnn import create_model

batch_size = 32
epochs = 30
optimizer_name = "adam"  # ou "sgd"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = create_dataloaders(batch_size=batch_size)
model = create_model(num_classes=6).to(device)
optimizer = create_optimizer(optimizer_name, model)
```

**Observações:** os CSVs e diretórios são definidos em `create_dataloaders()`; execute o script na raiz do projeto. O código atual considera `dataset/treino.csv`, `dataset/teste.csv`, `dataset/treino/` e `dataset/teste/`, com colunas `filename` e `label` e classes de 0 a 5.

## 4. Treinar e salvar

Acrescente ao mesmo script:

```python
from pathlib import Path

model = train_model(model, train_loader, optimizer, device, epochs=epochs)

Path("results").mkdir(exist_ok=True)
torch.save(model.state_dict(), f"results/minha_cnn_original_{optimizer_name}.pth")
```

Execute:

```bash
python main.py
```

Para comparar **Adam** e **SGD**, execute novamente com `optimizer_name = "sgd"`, **criando uma nova instância do modelo** em cada execução. Use nomes de arquivo diferentes para não sobrescrever os pesos.

## 5. Testar

O `test_loader` já está disponível. **A função compartilhada de teste ainda não foi implementada** na etapa atual do projeto; portanto, não existe uma chamada pronta a ser reutilizada para calcular acurácia, precisão, recall e F1. Quando `training/test.py` estiver integrado, todos devem utilizar a mesma função e as mesmas métricas.

## 6. Padrão de experimentos

| Entrada | Otimizadores |
|---|---|
| Original | Adam e SGD |
| Contrast stretching | Adam e SGD |
| Equalização de histograma | Adam e SGD |

**Por enquanto, somente a entrada original está implementada.** Os pré-processamentos serão integrados posteriormente.

**Para manter a comparação:** use os mesmos dados, transformações, número de épocas e definição de métricas; reinicialize a CNN em cada execução. Os valores `batch_size=32` e `epochs=30` são escolhas do projeto, não configurações confirmadas pelo artigo.
