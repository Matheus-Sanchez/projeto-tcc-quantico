# Relatório do treinamento quântico com Qiskit

Leitura: 09/10/2026 às 22h16, America/Sao_Paulo. Nenhum processo de treinamento estava ativo na leitura. Este relatório registra uma leitura pontual.

**9 concluídos, 0 em execução e 0 aguardando.** A execução terminou em 09/10/2026 às 04h31, com treino, teste e estudo de ruído concluídos nos nove datasets. Não há datasets aguardando na fila. Os nove datasets tiveram suas features validadas antes do primeiro treinamento.

| Dataset | Etapa | Épocas confirmadas | Melhor época | Acurácia de teste | Macro-F1 de teste |
| --- | --- | --- | --- | --- | --- |
| MNIST | Concluído: treino, teste e estudo de ruído | 100/100 | 13 | 99,30% | 99,30% |
| Fashion-MNIST | Concluído: treino, teste e estudo de ruído | 100/100 | 5 | 91,87% | 91,89% |
| KMNIST | Concluído: treino, teste e estudo de ruído | 100/100 | 30 | 98,65% | 98,65% |
| EMNIST Balanced | Concluído: treino, teste e estudo de ruído | 100/100 | 40 | 88,52% | 88,47% |
| CIFAR-10 | Concluído: treino, teste e estudo de ruído | 100/100 | 12 | 73,26% | 73,32% |
| CIFAR-100 Coarse | Concluído: treino, teste e estudo de ruído | 100/100 | 23 | 47,06% | 47,12% |
| SVHN | Concluído: treino, teste e estudo de ruído | 100/100 | 9 | 93,02% | 92,63% |
| GTSRB | Concluído: treino, teste e estudo de ruído | 100/100 | 99 | 99,64% | 99,59% |
| FER2013 | Concluído: treino, teste e estudo de ruído | 100/100 | 96 | 50,85% | 48,52% |

## CIFAR: novos resultados de teste

**CIFAR-10** terminou em 06/10/2026 às 21h33, após 14h01min. O checkpoint da época 12 foi selecionado pelo macro-F1 de validação de 73,35%. Nos 9.000 exemplos de teste, a acurácia foi **73,26%** e o macro-F1 foi **73,32%**. Foram concluídas 100 épocas e 64 condições de ruído. Na última época, a diferença entre acurácia de treino e validação foi de 26,85 pontos percentuais.

**CIFAR-100 Coarse** terminou em 07/10/2026 às 11h45, após 14h12min. O checkpoint da época 23 foi selecionado pelo macro-F1 de validação de 48,43%. Nos 9.000 exemplos de teste, a acurácia foi **47,06%** e o macro-F1 foi **47,12%**. Foram concluídas 100 épocas e 64 condições de ruído. Na última época, a diferença entre acurácia de treino e validação foi de 35,77 pontos percentuais.

## EMNIST Balanced concluído

A execução terminou em 06/10/2026 às 07h32, após 31h14min. O checkpoint da época 40 foi selecionado pelo macro-F1 de validação de 88,85%. No conjunto completo de 19.740 exemplos de teste, a acurácia foi **88,52%** e o macro-F1 foi **88,47%**. Foram salvas predições, métricas por classe, matriz de confusão e 64 condições de ruído.

## SVHN, GTSRB e FER2013: resultados finais

**SVHN** terminou em 08/10/2026 às 10h55, após 23h09min. O checkpoint da época 9 foi escolhido pelo macro-F1 de validação de 92,59%. Nos 14.893 exemplos de teste, a acurácia foi **93,02%** e o macro-F1 foi **92,63%**. Foram concluídas 100 épocas e 64 condições de ruído.

**GTSRB** terminou em 08/10/2026 às 20h08, após 9h14min. O checkpoint da época 99 foi escolhido pelo macro-F1 de validação de 99,75%. Nos 5.876 exemplos de teste, a acurácia foi **99,64%** e o macro-F1 foi **99,59%**. Foram concluídas 100 épocas e 64 condições de ruído.

**FER2013** terminou em 09/10/2026 às 04h31, após 8h23min. O checkpoint da época 96 foi escolhido pelo macro-F1 de validação de 50,23%. Nos 5.382 exemplos de teste, a acurácia foi **50,85%** e o macro-F1 foi **48,52%**. Foram concluídas 100 épocas e 64 condições de ruído.

No FER2013, a última época registrou treino 92,56% e validação 51,34%, diferença de 41,22 pontos percentuais. A acurácia mais alta do treino não se mantém na validação. O teste usa o checkpoint da época 96, selecionado pela validação.

## Dados coletados

- 900 registros de épocas completas, com loss, acurácia, macro-F1, métricas por classe e duração.
- 348.200 JSONs de lotes de treino e 75.300 de validação; quatro registros de circuito por arquivo, totalizando 1.694.000 registros. Todos os lotes pertencem a épocas completas.
- 107.796 amostras de telemetria (CPU, RAM e disco).
- Checkpoints, sondas de estados, gradientes, alterações de pesos, tempos por circuito e contagem de avaliações lógicas nos 9 treinamentos iniciados.
- Predições de teste, métricas por classe e matrizes de confusão nos 9 concluídos, cobrindo 95.389 exemplos de teste.
- 576 condições do estudo de ruído/shots: 64 por dataset concluído.

| Dataset | JSONs de treino | JSONs de validação | Sondas | Condições de ruído | Predições de teste |
| --- | --- | --- | --- | --- | --- |
| MNIST | 38.300 | 8.300 | 102 | 64 | Salvas |
| Fashion-MNIST | 38.300 | 8.300 | 102 | 64 | Salvas |
| KMNIST | 38.300 | 8.300 | 102 | 64 | Salvas |
| EMNIST Balanced | 72.000 | 15.500 | 102 | 64 | Salvas |
| CIFAR-10 | 32.900 | 7.100 | 102 | 64 | Salvas |
| CIFAR-100 Coarse | 32.900 | 7.100 | 102 | 64 | Salvas |
| SVHN | 54.300 | 11.700 | 102 | 64 | Salvas |
| GTSRB | 21.500 | 4.700 | 102 | 64 | Salvas |
| FER2013 | 19.700 | 4.300 | 102 | 64 | Salvas |

## Tempo e telemetria

| Dataset | Tempo da sessão | Amostras | Pico RAM da árvore | Última amostra |
| --- | --- | --- | --- | --- |
| MNIST | 16h22min | 11.742 | 2,80 GiB | 03/10/2026 às 14h42 |
| Fashion-MNIST | 16h39min | 11.952 | 2,92 GiB | 04/10/2026 às 07h21 |
| KMNIST | 16h56min | 12.157 | 2,61 GiB | 05/10/2026 às 00h18 |
| EMNIST Balanced | 31h14min | 22.418 | 3,16 GiB | 06/10/2026 às 07h32 |
| CIFAR-10 | 14h01min | 10.061 | 2,44 GiB | 06/10/2026 às 21h33 |
| CIFAR-100 Coarse | 14h12min | 10.197 | 2,04 GiB | 07/10/2026 às 11h45 |
| SVHN | 23h09min | 16.621 | 2,67 GiB | 08/10/2026 às 10h55 |
| GTSRB | 9h14min | 6.628 | 2,18 GiB | 08/10/2026 às 20h08 |
| FER2013 | 8h23min | 6.020 | 2,21 GiB | 09/10/2026 às 04h31 |

## Estudo de ruído

Escalas 0, 0,5, 1 e 2; leitura exata e 256/1.024/4.096 shots; cinco seeds (42 a 46) por condição finita. O estudo reutiliza o melhor checkpoint e simula todo o teste localmente.

Na escala 2 com 256 shots (queda negativa indica aumento da acurácia):

| Dataset | Acurácia média | Mínimo | Máximo | Queda média (p.p.) |
| --- | --- | --- | --- | --- |
| MNIST | 99,27% | 99,26% | 99,28% | 0,032 |
| Fashion-MNIST | 91,80% | 91,73% | 91,88% | 0,067 |
| KMNIST | 98,60% | 98,56% | 98,65% | 0,048 |
| EMNIST Balanced | 88,40% | 88,31% | 88,48% | 0,119 |
| CIFAR-10 | 73,02% | 72,91% | 73,19% | 0,231 |
| CIFAR-100 Coarse | 46,63% | 46,51% | 46,89% | 0,427 |
| SVHN | 92,95% | 92,89% | 92,99% | 0,070 |
| GTSRB | 99,65% | 99,63% | 99,68% | -0,003 |
| FER2013 | 50,61% | 50,43% | 50,87% | 0,245 |

Mínimo e máximo descrevem as cinco seeds de amostragem, sem representar um intervalo de confiança. Essas seeds não são repetições independentes do treinamento.

## Features disponíveis

| Dataset | Classes | Treino | Validação | Teste |
| --- | --- | --- | --- | --- |
| MNIST | 10 | 48.999 | 10.503 | 10.498 |
| Fashion-MNIST | 10 | 49.000 | 10.500 | 10.500 |
| KMNIST | 10 | 49.000 | 10.500 | 10.500 |
| EMNIST Balanced | 47 | 92.120 | 19.740 | 19.740 |
| CIFAR-10 | 10 | 42.000 | 9.000 | 9.000 |
| CIFAR-100 Coarse | 20 | 42.000 | 9.000 | 9.000 |
| SVHN | 10 | 69.502 | 14.894 | 14.893 |
| GTSRB | 43 | 27.489 | 5.905 | 5.876 |
| FER2013 | 7 | 25.121 | 5.384 | 5.382 |

São 636.046 exemplos entre todos os splits, cada um com 128 valores float32. Nos 9 datasets iniciados, as estatísticas salvas registram zero NaN e zero infinito.

## Protocolo e interpretação

Cabeça: features128 → Dense(128, ReLU) → Dense(20, linear) → quatro circuitos de cinco qubits → logits. São 180 parâmetros quânticos e 19.482 parâmetros treináveis para dez classes. Qiskit Aer local em CPU, quatro workers persistentes, Adam, learning rate 0,0003, batch 128, seed 42 e 100 épocas. Sem parada antecipada ou redução automática da taxa de aprendizado.

Os checkpoints usados no teste foram selecionados pelo maior macro-F1 de validação: época 13 (MNIST), época 5 (Fashion-MNIST), época 30 (KMNIST), época 40 (EMNIST Balanced), época 12 (CIFAR-10), época 23 (CIFAR-100 Coarse), época 9 (SVHN), época 99 (GTSRB), época 96 (FER2013). No Fashion-MNIST, a última época tem treino 99,95% e validação 91,97%; a avaliação final utiliza a época 5.

Todas as 900 épocas planejadas foram concluídas (100%). Não há treino, teste ou estudo de ruído pendente.

A referência clássica salva nos 9 runs é inelegível: faltam status concluído e métricas de teste naquela referência. Este relatório não estabelece ganho quântico sobre o clássico. Há um único seed de treinamento por dataset.

## Fontes e verificações

Arquivos locais em outputs/quantum-parallel-128-20/runs: matrix.json; dataset/qiskit/seed-42/{status.json, checkpoints/state.json, history.csv, test_metrics.json, result.json, config.json, feature_statistics.json}; diagnostics/{batches, probes, noise-study}; telemetry/*/samples.jsonl. Features em outputs/classic-dense20/*/runs/*/artifacts/features128.

O coletor está em collect_evidence.py, neste diretório. Os 54 cabeçalhos NPY foram conferidos. Históricos e melhor macro-F1 foram reconciliados com os checkpoints; métricas de teste foram reconciliadas com os 9 resultados. Configurações e estatísticas de features foram conferidas nos 9 runs iniciados. Todos os lotes pertencem a épocas completas. O relatório interativo mantém as fontes e definições nos menus de cada componente.
