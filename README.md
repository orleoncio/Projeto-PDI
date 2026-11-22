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

## Treinamento e Replicação da VGG-16

A [VGG-16](models/vgg16.py) (Simonyan & Zisserman, 2014) está disponível em `models/vgg16.py`, configurada com as 16 camadas com pesos treináveis (13 convolucionais com filtros 3x3 e 3 totalmente conectadas) ajustadas para as 6 classes de células HEp-2 e sem pesos pré-treinados (`weights=None`), em total conformidade com a metodologia do artigo.

O treinamento e a avaliação podem ser executados via terminal ou no Google Colab (`treinamento_vgg16_colab.ipynb`):

```bash
# Treinamento com SGD (Imagem Original)
python main_vgg16.py --optimizer sgd --epochs 30 --batch-size 32

# Treinamento com Adam (Imagem Original)
python main_vgg16.py --optimizer adam --epochs 30 --batch-size 32
```

### Resultados Experimentais Obtidos (Confrontação com o Artigo - Tabela 3)

| Otimizador | Métrica | Artigo Oficial (%) | Nosso Treino (%) | Diferença ($\Delta$) | Diagnóstico Metodológico |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **SGD** | **Accuracy** | 98.16% | **97.98%** | **-0.18%** | Replicação fiel (Data Augmentation $D_4$ + fine-tuning) |
| **SGD** | **F1-score** | 98.05% | **97.64%** | **-0.41%** | Equilíbrio entre precision e recall |
| **SGD** | **Precision** | 98.14% | **97.56%** | **-0.58%** | |
| **SGD** | **Recall** | 97.98% | **97.73%** | **-0.25%** | |
| **ADAM** | **Accuracy** | 19.96% | **21.73%** | +1.77% | **Replicação perfeita do colapso do artigo** |
| **ADAM** | **F1-score** | 5.55% | **5.95%** | +0.40% | Mínimo local severo / subajuste |
| **ADAM** | **Precision** | 3.33% | **3.62%** | +0.29% | |
| **ADAM** | **Recall** | 16.67% | **16.67%** | **0.00%** | Predição de classe única ($1/6 = 16.67\%$) |

> 📄 Para uma análise aprofundada sobre a dinâmica do treinamento, a influência do otimizador e o impacto de *Transfer Learning*, consulte o [Relatório Técnico da VGG-16](docs/descobertas_vgg16_artigo.md).

## Treinamento da SqueezeNet

A [SqueezeNet 1.1 do torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_1.html) está disponível em `models/squeezenet.py`, com seis classes e sem pesos pré-treinados. O treinamento reutiliza o dataset, o otimizador e o laço de treino compartilhados, seguindo o [guia de uso das CNNs](guia_uso_cnns.md).

Na raiz do projeto, compare Adam e SGD nos três cenários, mantendo SqueezeNet 1.1, 30 épocas e lote 32:

```bash
python main_squeezenet.py --preprocessing none --optimizer adam --epochs 30 --batch-size 32
python main_squeezenet.py --preprocessing none --optimizer sgd --epochs 30 --batch-size 32
python main_squeezenet.py --preprocessing contrast_expansion --optimizer adam --epochs 30 --batch-size 32
python main_squeezenet.py --preprocessing contrast_expansion --optimizer sgd --epochs 30 --batch-size 32
python main_squeezenet.py --preprocessing histogram_equalization --optimizer adam --epochs 30 --batch-size 32
python main_squeezenet.py --preprocessing histogram_equalization --optimizer sgd --epochs 30 --batch-size 32
```

O script seleciona GPU quando disponível, imprime accuracy, precision, recall e F1-score no conjunto de teste e salva cada checkpoint em `results/squeezenet_<cenário>_<otimizador>.pth`. `contrast_expansion` aplica o aumento de contraste já implementado no projeto (fator 1.5). Também aceita `--eval-only` para avaliar um checkpoint existente.

No Colab, ative a GPU, disponibilize o projeto e o dataset, entre na raiz do projeto e execute os comandos com `!python -u`. O notebook com as seis execuções está em [treinamento_squeezenet_colab.ipynb](treinamento_squeezenet_colab.ipynb); as métricas já medidas e o estado das rodadas estão em [resultados_squeezenet.md](docs/resultados_squeezenet.md).

## Trabalhos Futuros e Melhorias

Além da replicação fiel do artigo original, o escopo futuro deste projeto inclui:
* Explorar pipelines de pré-processamento avançados para aprimorar a extração de características.

---
*Criado para fins de pesquisa, replicação e aprimoramento contínuo em aprendizado de máquina aplicado à saúde.*
