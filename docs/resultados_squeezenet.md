# Experimentos da SqueezeNet 1.1 no HEp-2

## Configuração planejada

O experimento compara seis combinações, todas treinadas do zero por 30 épocas, com lote 32, seis classes e as mesmas divisões do dataset (10.877 imagens de treino e 2.720 de teste):

| Entrada | Adam | SGD |
|---|---:|---:|
| Original | Planejado | Planejado |
| Expansão de contraste | Planejado | Planejado |
| Equalização de histograma | Planejado | Planejado |

O pipeline usa as transformações de treino já definidas no projeto, incluindo flips horizontal/vertical e rotação em múltiplos de 90°. A avaliação usa as imagens determinísticas do conjunto de teste e reporta accuracy, precision macro, recall macro e F1 macro. Adam usa os padrões do projeto (lr 0,001); SGD usa lr 0,01 e momentum 0,9.

## Medição disponível neste ambiente

Há um checkpoint anterior em `results/squeezenet_original_adam.pth`. Ele foi avaliado no conjunto de teste sem novo treinamento nesta execução:

| Cenário | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Original + Adam (checkpoint existente) | 74,04% | 63,86% | 71,58% | 66,56% |

Este resultado é uma avaliação do checkpoint preexistente e não deve ser tratado como uma nova rodada reproduzida. As outras cinco combinações ainda não têm métricas.

## Execução das seis rodadas

Use o notebook [treinamento_squeezenet_colab.ipynb](../treinamento_squeezenet_colab.ipynb) numa sessão Colab com GPU. Cada execução imprime as quatro métricas e salva os pesos em `results/squeezenet_<cenário>_<otimizador>.pth`.

O ambiente local desta execução não tem CUDA/GPU. Uma tentativa curta de medir uma época na CPU não terminou dentro de aproximadamente dois minutos; por isso, não iniciei seis treinamentos de 30 épocas nesse hardware. Não extrapolei nem inventei métricas para as rodadas pendentes.
