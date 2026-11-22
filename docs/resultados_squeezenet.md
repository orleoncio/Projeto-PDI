# Consolidação Experimental dos Resultados da SqueezeNet

Este documento reúne os seis cenários da **SqueezeNet** no dataset HEp-2 e compara as métricas dos checkpoints exportados do Colab com os números da Tabela 3 do artigo *Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis*.

## 1. Configuração e origem dos resultados

O projeto instancia a **SqueezeNet 1.1** do TorchVision, treinada do zero para seis classes. O artigo reporta **SqueezeNet 1.0**; portanto, os valores publicados são referência contextual, não uma comparação da mesma arquitetura.


## 2. Comparação geral: artigo vs. checkpoints do Colab

As diferenças são calculadas como resultado do projeto menos o valor de referência, em pontos percentuais (p.p.).

| Entrada | Otimizador | Métrica | Artigo, SqueezeNet 1.0 (%) | Projeto, SqueezeNet 1.1 (%) | Δ (p.p.) | Observação |
| :--- | :---: | :---: | ---: | ---: | ---: | :--- |
| **Original** | **Adam** | Accuracy<br>Precision<br>Recall<br>F1-score | 93,90<br>93,47<br>93,45<br>93,43 | **92,35**<br>**91,78**<br>**92,84**<br>**92,15** | -1,55<br>-1,69<br>-0,61<br>-1,28 | Resultado próximo à referência, com maior recall que precision. |
| **Original** | **SGD** | Accuracy<br>Precision<br>Recall<br>F1-score | 96,40<br>96,82<br>95,66<br>96,18 | **89,93**<br>**89,78**<br>**90,96**<br>**90,25** | -6,47<br>-7,04<br>-4,70<br>-5,93 | Todas as métricas ficaram abaixo da referência. |
| **Contraste** | **Adam** | Accuracy<br>Precision<br>Recall<br>F1-score | 92,68<br>92,75<br>91,65<br>92,09 | **91,32**<br>**92,08**<br>**90,73**<br>**91,05** | -1,36<br>-0,67<br>-0,92<br>-1,04 | Resultado próximo à referência; precision quase igual. |
| **Contraste** | **SGD** | Accuracy<br>Precision<br>Recall<br>F1-score | 96,47<br>96,39<br>96,33<br>96,35 | **89,60**<br>**91,21**<br>**89,41**<br>**90,14** | -6,87<br>-5,18<br>-6,92<br>-6,21 | Precision é a métrica mais próxima da referência; as demais ficaram cerca de 6–7 p.p. abaixo. |
| **Equalização** | **Adam** | Accuracy<br>Precision<br>Recall<br>F1-score | 86,62<br>87,79<br>86,48<br>86,53 | **87,43**<br>**88,48**<br>**87,24**<br>**87,78** | +0,81<br>+0,69<br>+0,76<br>+1,25 | Superou ligeiramente os quatro valores de referência. |
| **Equalização** | **SGD** | Accuracy<br>Precision<br>Recall<br>F1-score | 96,29<br>96,37<br>95,78<br>96,05 | **80,00**<br>**67,38**<br>**71,19**<br>**68,79** | -16,29<br>-28,99<br>-24,59<br>-27,26 | É o pior cenário medido; a queda é especialmente forte em precision e F1. |

## 3. Tabela resumida para apresentação

| Pré-processamento | Otimizador | Accuracy (%) | F1 macro (%) |
| :--- | :---: | ---: | ---: |
| Original | Adam | **92,35** | **92,15** |
| Original | SGD | 89,93 | 90,25 |
| Contraste | Adam | 91,32 | 91,05 |
| Contraste | SGD | 89,60 | 90,14 |
| Equalização | Adam | 87,43 | 87,78 |
| Equalização | SGD | 80,00 | 68,79 |

## 4. Análise dos resultados

- **Adam teve o melhor resultado geral.** Na imagem original, obteve 92,35% de accuracy e 92,15% de F1. No contraste, ficou próximo, com 91,32% e 91,05%. A equalização reduziu a accuracy para 87,43%, mas o resultado ainda ficou próximo da referência publicada para esse cenário.
- **O SGD teve resultados parecidos com Adam em original e contraste**, com F1 de 90,25% e 90,14%, respectivamente. Em ambos, as métricas ficaram abaixo dos números do artigo para SqueezeNet 1.0.
- **Equalização com SGD destoou dos outros cinco cenários.** A accuracy foi 80,00%, mas o F1 macro caiu para 68,79%; precision macro (67,38%) e recall macro (71,19%) também foram baixos. O checkpoint indica desempenho desigual no agregado, mas, sem matriz de confusão ou métricas por classe, não é possível identificar quais classes concentram os erros.
- **A expansão de contraste não melhorou o resultado de Adam sobre a imagem original** nesta execução: a accuracy caiu 1,03 p.p. e o F1 caiu 1,10 p.p. Com SGD, contraste e imagem original tiveram resultados próximos.
- A diferença absoluta média frente à referência é **5,56 p.p. em accuracy** e **7,16 p.p. em F1** nos seis cenários. Esses valores devem ser lidos considerando a diferença entre SqueezeNet 1.0 e 1.1 e a ausência dos logs de treino no ZIP.
