# Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis

Este repositório contém a implementação dos experimentos apresentados no artigo científico **"Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis"**. O objetivo principal é reproduzir a metodologia original e, futuramente, explorar melhorias para alcançar resultados superiores.

## Sobre o Projeto

O diagnóstico baseado em imagens médicas utilizando aprendizado de máquina e aprendizado profundo depende fortemente da qualidade dos dados de entrada e da eficiência dos algoritmos de otimização. Este projeto implementa o fluxo experimental descrito no estudo de referência para analisar o impacto dessas variáveis.

## Artigo Base

* **Título:** Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis
* **Descrição:** Estudo sobre a influência de técnicas de pré-processamento de imagens e otimizadores no desempenho de modelos voltados para a detecção de doenças autoimunes.

## Estrutura do Repositório

```text
├── data/           # Datasets utilizados nos experimentos
├── src/            # Código-fonte principal (pré-processamento, modelos, treino)
└── README.md       # Documentação do repositório
```

## Treinamento da SqueezeNet

A [SqueezeNet 1.1 do torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_1.html) está disponível em `models/squeezenet.py`, com seis classes e sem pesos pré-treinados. O treinamento reutiliza o dataset, o otimizador e o laço de treino compartilhados, seguindo o [guia de uso das CNNs](guia_uso_cnns.md).

Na raiz do projeto, com as mesmas dependências utilizadas pela ResNet-50, execute:

```bash
python main_squeezenet.py
```

O script usa 30 épocas, lote 32 e ADAM, selecionando GPU quando disponível. Ao finalizar, salva os pesos em `results/squeezenet_original_adam.pth`. Para usar SGD, altere `optimizer_name` em `main_squeezenet.py` para `"sgd"`; o arquivo será `results/squeezenet_original_sgd.pth`. Cada execução cria um novo modelo.

No Colab, ative a GPU, disponibilize o projeto e o dataset, entre na raiz do projeto e execute `!python main_squeezenet.py`.

## Trabalhos Futuros e Melhorias

Além da replicação fiel do artigo original, o escopo futuro deste projeto inclui:
* Explorar pipelines de pré-processamento avançados para aprimorar a extração de características.

---
*Criado para fins de pesquisa, replicação e aprimoramento contínuo em aprendizado de máquina aplicado à saúde.*
