# Resultados coletados: treino clássico Dense20 com vetores congelados

**Data da consolidação:** 30/09/2026  |  **Situação:** 9 de 9 treinos clássicos concluídos

## Resumo

- Os nove conjuntos foram treinados por **100 épocas**, com seed 42, e avaliados no conjunto de teste separado: **95.389 exemplos de teste** no total. São tarefas distintas; o total é cobertura, não uma acurácia agrupada.
- Média simples entre datasets: **82.51% de acurácia**, **82.18% de macro-F1** e **96.06% de macro-AUC OVR**. Cada dataset tem o mesmo peso nessa média, independentemente do tamanho.
- O melhor resultado de teste foi GTSRB (acurácia 99.66%; macro-F1 99.63%). O menor macro-F1 foi CIFAR-100 coarse (47.56%).
- TensorFlow foi detectado somente em CPU nesta execução (`macOS-15.5-arm64`, Python 3.12.3, TensorFlow 2.18.1).

## Resultados por dataset

Métricas de classificação são do conjunto de teste. Precision, recall, F1 e AUC são médias macro; AUC é one-vs-rest. `Loss` é a loss compilada registrada na avaliação. “Melhor época” é a época de maior macro-F1 na validação, usada para selecionar `best.keras`.

| Dataset | Classes | n treino / validação / teste | Acurácia | Acurácia balanceada | Macro precision | Macro recall | Macro-F1 | Macro-AUC OVR | Loss teste | Melhor época | Macro-F1 validação |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 10 | 42.000 / 9.000 / 9.000 | 73.32% | 73.32% | 73.50% | 73.32% | 73.39% | 95.31% | 1.1202 | 9/100 | 73.59% |
| CIFAR-100 coarse | 20 | 42.000 / 9.000 / 9.000 | 47.41% | 47.41% | 47.84% | 47.41% | 47.56% | 89.30% | 2.1101 | 24/100 | 48.99% |
| EMNIST Balanced | 47 | 92.120 / 19.740 / 19.740 | 88.69% | 88.69% | 88.70% | 88.69% | 88.64% | 99.65% | 0.3627 | 26/100 | 88.97% |
| Fashion-MNIST | 10 | 49.000 / 10.500 / 10.500 | 92.05% | 92.05% | 92.05% | 92.05% | 92.05% | 99.30% | 0.3030 | 6/100 | 92.52% |
| FER2013 | 7 | 25.121 / 5.384 / 5.382 | 50.58% | 46.64% | 50.11% | 46.64% | 47.90% | 81.67% | 1.9238 | 10/100 | 50.02% |
| GTSRB | 43 | 27.489 / 5.905 / 5.876 | 99.66% | 99.69% | 99.58% | 99.69% | 99.63% | 99.996% | 0.0201 | 73/100 | 99.76% |
| KMNIST | 10 | 49.000 / 10.500 / 10.500 | 98.65% | 98.65% | 98.65% | 98.65% | 98.65% | 99.92% | 0.0838 | 30/100 | 98.81% |
| MNIST | 10 | 48.999 / 10.503 / 10.498 | 99.23% | 99.22% | 99.23% | 99.22% | 99.23% | 99.98% | 0.0359 | 19/100 | 99.28% |
| SVHN | 10 | 69.502 / 14.894 / 14.893 | 92.98% | 92.53% | 92.61% | 92.53% | 92.57% | 99.44% | 0.2809 | 6/100 | 92.54% |

## Configuração executada

- Entrada: vetores `float32` de 128 dimensões, extraídos após `block5_pool` + `GlobalAveragePooling2D`; os manifests registram a CNN congelada sem alterações antes/depois da extração.
- Ponta treinável: `128 → Dense(128, ReLU) → Dense(20, tanh) → Dense(n_classes, logits)`, usando os mesmos parâmetros de treinamento nos nove datasets.
- Adam, learning rate `0.0003`, batch size `128`, cross-entropy, seed `42`, precisão `float32`; 100 épocas fixas, sem early stopping nem scheduler. `best.keras` foi selecionado por macro-F1 de validação.
- Não houve nova normalização, scaler, PCA, seleção de features ou augmentation dos vetores nessa etapa.
- Vetores consumidos: **445.231 treino + 95.426 validação + 95.389 teste = 636.046** exemplos, somados nos nove datasets.
- Soma dos tempos de época registrados: **194.9 s**; soma dos tempos de parede registrados por execução: **284.7 s**. Esses tempos são somados por dataset, não medem uso do pipeline de extração de features.

## Leitura dos resultados e limites

- As diferenças entre datasets são grandes: MNIST, KMNIST e GTSRB têm macro-F1 acima de 98%; CIFAR-10, CIFAR-100 coarse e FER2013 têm desempenho bem menor. Isso descreve esta ponta e estes vetores; não mede a qualidade isolada da CNN nem demonstra vantagem sobre outra arquitetura.
- A diferença entre acurácia de treino na melhor época e acurácia de teste foi maior em FER2013 (36.78 p.p.), CIFAR-100 coarse (31.24 p.p.) e CIFAR-10 (22.73 p.p.). É um sinal observado de ajuste desigual entre treino e teste, não uma prova isolada de sua causa.
- Cada dataset teve uma execução, com uma única seed (42). Não há intervalos de confiança, repetição entre seeds nem testes estatísticos; a média entre datasets é descritiva, não uma métrica de um dataset combinado.
- **Ressalva específica do KMNIST:** o relatório de andamento do projeto documenta que a CNN congelada da fase 1 foi treinada usando os 70.000 exemplos combinados e que, no split refeito da fase 2, 5.284/9.000 exemplos de validação e 5.700/9.000 de teste já tinham sido vistos por aquela CNN. Portanto, o resultado KMNIST mede a cabeça sobre embeddings com sobreposição de exposição do encoder e não uma avaliação independente da CNN congelada. Esse achado não foi extrapolado para os outros datasets.
- Os arquivos revisados não comprovam a independência entre o treino original de cada CNN congelada e seus splits de avaliação para todos os demais datasets.

## QCNN no disco

Os diretórios QCNN não têm avaliação final coletada. A execução `phase2-full-kmnist-50ep-20260928-174022` ainda está marcada como `running` no último `run-status.json` e tem histórico até a época 32/50: melhor validação macro-F1 99,13% na época 30; época 32 em 99,03%. Não há artefato de métricas finais de teste. A execução `phase2-full-kmnist-50ep-20260929-214413` também está marcada como `running`, mas não tem histórico de época nem resultado final nos artefatos vistos. Por isso elas não entram na tabela dos nove treinos completos.

## Arquivos de evidência

- Agregado: [`classical_128_20_results.csv`](classical_128_20_results.csv).
- Por dataset: `outputs/classical-128-20-mac/<dataset>/seed-42/`: `status.json`, `config.json`, `timing.json`, `history.csv`, `test_metrics.json`, `classification_report.json`, `predictions.csv`, `confusion_matrix.npy`, `best.keras` e `final.keras`.
- Vetores e metadados do encoder: `outputs/classic-dense20/<dataset>/runs/<run>/artifacts/features128/`, especialmente `manifest.json` e os arquivos `.npy` de cada split.
- O relatório de andamento QCNN citado acima: [`RELATORIO_ANDAMENTO_TREINAMENTO_QUANTICO.md`](../../RELATORIO_ANDAMENTO_TREINAMENTO_QUANTICO.md).
- Preflight do ambiente: [`preflight.json`](preflight.json).
