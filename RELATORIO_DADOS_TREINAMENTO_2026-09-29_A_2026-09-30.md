# Relatório completo dos dados de treinamento — 29 a 30/09/2026

**Corte da consolidação:** 30/09/2026 (horário de Brasília, UTC−03:00).
**Janela solicitada:** 29/09 a 30/09/2026. Os horários abaixo foram convertidos para BRT a partir dos registros ISO originais.
**Escopo principal:** nove treinamentos independentes da ponta Dense de 128→128→20→classes, executados no Mac em nove datasets, usando vetores congelados pré-computados em WSL2.

## Resumo executivo

- **9 de 9** treinamentos Dense concluídos; **900 épocas** registradas (100 por dataset), seed 42.
- Foram consumidos **636.046 vetores** de 128 características: 445.231 treino, 95.426 validação e 95.389 teste. O teste final cobriu 95.389 exemplos nos nove problemas separados.
- Média simples entre datasets: acurácia **82,51%**, macro-F1 **82,18%** e macro-AUC OVR **96,06%**. Cada dataset recebe o mesmo peso; isso não é uma métrica de um dataset combinado.
- Melhor macro-F1 de teste: GTSRB. Menor macro-F1 de teste: CIFAR-100 coarse.
- A pré-verificação registrou somente CPU TensorFlow. A telemetria não registrou utilização de GPU; os nove treinos Dense foram executados sem evidência de aceleração por GPU.
- O diretório de resultados Dense contém 171 arquivos (aprox. 57,8 MiB), incluindo pesos, curvas completas, métricas, previsões, logits, matrizes e telemetria.

## Linha do tempo observada

| Etapa | Janela em BRT | Estado registrado |
| --- | --- | --- |
| Extração dos vetores 128-D | 29/09/2026 18:01:28 a 29/09/2026 18:04:29 (WSL2/Linux) | 9/9 manifests como completed; encoder marcado como inalterado após a extração |
| Treinamentos Dense | 29/09/2026 22:36:52 a 29/09/2026 22:41:39 (Mac/macOS) | 9/9 completed; 100 épocas em cada histórico |

Os nove treinamentos Dense no Mac foram concluídos em 29/09 à noite. Eles consumiram vetores extraídos anteriormente em WSL2; portanto, a execução no Mac corresponde ao treino das pontas Dense e à avaliação final, não à extração dos embeddings. A consolidação e a gravação deste relatório ocorreram em 30/09; não houve novas épocas Dense nessa data.

## Entrada e proveniência dos dados

Cada cabeça recebeu um vetor de 128 valores float32 por exemplo. Os vetores foram extraídos em WSL2/Linux, conforme o campo de plataforma nos registros de telemetria associados aos nove manifests; depois, esses vetores pré-computados foram usados nos treinamentos Dense no Mac. O manifesto define-os como saída de block5_pool seguida de GlobalAveragePooling2D, produzidos pela CNN congelada em modo de inferência. Os nove manifests registram backbone_unchanged: true; as estatísticas de features registram zero NaN e zero infinito em todos os splits.

Na extração e no treinamento da ponta, a configuração registra augmentation desativada e nenhuma seleção de features, PCA, scaler ou normalização adicional dos vetores. Os manifests apontam para checkpoints do caminho controlled-augmentation05-batch-activation e para execuções de origem identificadas como dense20_fp32_aug05; isso é a proveniência do encoder, não augmentation aplicada de novo aos vetores. A normalização da entrada de imagem do encoder consta como unit_interval; isso não significa que os embeddings ficaram limitados a [0,1].

| Dataset | Classes | Treino | Validação | Teste |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 10 | 42.000 | 9.000 | 9.000 |
| CIFAR-100 coarse | 20 | 42.000 | 9.000 | 9.000 |
| EMNIST Balanced | 47 | 92.120 | 19.740 | 19.740 |
| Fashion-MNIST | 10 | 49.000 | 10.500 | 10.500 |
| FER2013 | 7 | 25.121 | 5.384 | 5.382 |
| GTSRB | 43 | 27.489 | 5.905 | 5.876 |
| KMNIST | 10 | 49.000 | 10.500 | 10.500 |
| MNIST | 10 | 48.999 | 10.503 | 10.498 |
| SVHN | 10 | 69.502 | 14.894 | 14.893 |
| **Total de vetores** | — | **445.231** | **95.426** | **95.389** |

As dimensões registradas em cada manifesto são [n_exemplos, 128] para os três arquivos de features; os rótulos são vetores int64. Os manifests guardam hashes SHA-256 para features, rótulos, índices, encoder e checkpoint de origem.

### Estatísticas dos embeddings

Média e desvio-padrão globais sobre os valores de cada split vêm de feature_statistics.json; a coluna final é o menor e o maior valor observado nos três splits daquele dataset.

| Dataset | Treino: média ± DP | Validação: média ± DP | Teste: média ± DP | Mínimo–máximo |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | -0,01290 ± 0,13621 | -0,01361 ± 0,13452 | -0,01346 ± 0,13478 | -0,27826 a 2,40688 |
| CIFAR-100 coarse | -0,02041 ± 0,13507 | -0,02074 ± 0,13366 | -0,02118 ± 0,13354 | -0,27843 a 2,67514 |
| EMNIST Balanced | -0,01244 ± 0,17228 | -0,01256 ± 0,17224 | -0,01265 ± 0,17205 | -0,27843 a 2,53716 |
| Fashion-MNIST | 0,01912 ± 0,24487 | 0,01882 ± 0,24486 | 0,01822 ± 0,24399 | -0,27841 a 2,78377 |
| FER2013 | -0,00422 ± 0,16729 | -0,00560 ± 0,16446 | -0,00477 ± 0,16670 | -0,27840 a 2,54262 |
| GTSRB | -0,00056 ± 0,18794 | -0,00088 ± 0,18749 | -0,00069 ± 0,18757 | -0,27239 a 1,67469 |
| KMNIST | 0,06409 ± 0,28605 | 0,06350 ± 0,28517 | 0,06351 ± 0,28505 | -0,27843 a 2,82021 |
| MNIST | 0,13229 ± 0,37577 | 0,13204 ± 0,37541 | 0,13181 ± 0,37498 | -0,27843 a 2,69377 |
| SVHN | 0,01328 ± 0,17491 | 0,01268 ± 0,17410 | 0,01286 ± 0,17441 | -0,27825 a 3,15782 |

## Protocolo efetivamente gravado

- Experimento: classical_128_relu_20_tanh_features128; um modelo separado para cada dataset.
- Seed 42; 100 épocas fixas; batch size 128; Adam; learning rate 0,0003; entropia cruzada esparsa com logits; precisão float32.
- Sem early stopping e sem scheduler. O checkpoint selecionado em cada manifesto é best.keras, pelo maior macro-F1 de validação; final.keras preserva o estado após a época 100.
- Arquitetura treinável: vetor de 128 → Dense(128, ReLU) → Dense(20) + tanh → Dense(n_classes, logits). O resumo salvo para MNIST registra 19.302 parâmetros treináveis; o total varia com o número de classes.
- A CNN de extração permaneceu congelada. Portanto, esta etapa mede o classificador Dense sobre embeddings pré-computados, não o fine-tuning da CNN.

## Métricas de validação e seleção do checkpoint

| Dataset | Melhor época | Macro-F1 validação no melhor checkpoint | Macro-F1 validação na época 100 |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 9/100 | 73,59% | 72,67% |
| CIFAR-100 coarse | 24/100 | 48,99% | 47,74% |
| EMNIST Balanced | 26/100 | 88,97% | 88,62% |
| Fashion-MNIST | 6/100 | 92,52% | 92,13% |
| FER2013 | 10/100 | 50,02% | 49,67% |
| GTSRB | 73/100 | 99,76% | 99,76% |
| KMNIST | 30/100 | 98,81% | 98,77% |
| MNIST | 19/100 | 99,28% | 99,26% |
| SVHN | 6/100 | 92,54% | 92,33% |

## Avaliação final no teste

As métricas abaixo vêm de test_metrics.json e classification_report.json, após seleção do checkpoint pela validação. Precision, recall e F1 são macro; AUC é one-vs-rest. Métricas em percentual estão arredondadas a duas casas decimais; a loss é a loss Keras da avaliação.

| Dataset | n teste | Acurácia | Acurácia balanceada | Macro-P | Macro-R | Macro-F1 | Macro-AUC OVR | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 9.000 | 73,32% | 73,32% | 73,50% | 73,32% | 73,39% | 95,31% | 1,1202 |
| CIFAR-100 coarse | 9.000 | 47,41% | 47,41% | 47,84% | 47,41% | 47,56% | 89,30% | 2,1101 |
| EMNIST Balanced | 19.740 | 88,69% | 88,69% | 88,70% | 88,69% | 88,64% | 99,65% | 0,3627 |
| Fashion-MNIST | 10.500 | 92,05% | 92,05% | 92,05% | 92,05% | 92,05% | 99,30% | 0,3030 |
| FER2013 | 5.382 | 50,58% | 46,64% | 50,11% | 46,64% | 47,90% | 81,67% | 1,9238 |
| GTSRB | 5.876 | 99,66% | 99,69% | 99,58% | 99,69% | 99,63% | 100,00% | 0,0201 |
| KMNIST | 10.500 | 98,65% | 98,65% | 98,65% | 98,65% | 98,65% | 99,92% | 0,0838 |
| MNIST | 10.498 | 99,23% | 99,22% | 99,23% | 99,22% | 99,23% | 99,98% | 0,0359 |
| SVHN | 14.893 | 92,98% | 92,53% | 92,61% | 92,53% | 92,57% | 99,44% | 0,2809 |

### Variação entre classes no teste

Para não esconder classes difíceis atrás das médias macro, a tabela mostra as classes de menor e maior F1 por dataset. O relatório por classe completo, com precision, recall, F1 e suporte para cada classe, está ligado na seção de artefatos.

| Dataset | Menor F1 por classe | Maior F1 por classe |
| --- | --- | --- |
| CIFAR-10 | classe 3: 52,99% (n=900) | classe 8: 84,62% (n=900) |
| CIFAR-100 coarse | classe 16: 26,89% (n=450) | classe 17: 75,63% (n=450) |
| EMNIST Balanced | classe 21: 55,42% (n=420) | classe 32: 98,94% (n=420) |
| Fashion-MNIST | classe 6: 77,88% (n=1.050) | classe 1: 98,90% (n=1.050) |
| FER2013 | classe 2: 33,29% (n=768) | classe 3: 71,83% (n=1.348) |
| GTSRB | classe 34: 97,67% (n=63) | classe 0: 100,00% (n=31) |
| KMNIST | classe 6: 97,87% (n=1.050) | classe 9: 99,29% (n=1.050) |
| MNIST | classe 9: 98,75% (n=1.044) | classe 0: 99,56% (n=1.035) |
| SVHN | classe 8: 90,57% (n=1.006) | classe 1: 95,11% (n=2.844) |

## Curvas de treino e custo computacional

As curvas completas de 100 épocas estão nos history.csv por dataset. A tabela resume o último registro de treino e o checkpoint da melhor época. A diferença entre treino e teste é um sinal descritivo de generalização; isoladamente, não identifica a causa.

| Dataset | Loss treino na época 100 | Acurácia treino na época 100 | Acurácia treino no melhor checkpoint | Melhor macro-F1 validação | Macro-F1 validação na época 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0,043338 | 98,72% | 96,05% | 73,59% | 72,67% |
| CIFAR-100 coarse | 0,527610 | 82,94% | 78,65% | 48,99% | 47,74% |
| EMNIST Balanced | 0,123390 | 94,98% | 93,78% | 88,97% | 88,62% |
| Fashion-MNIST | 0,005085 | 99,89% | 98,64% | 92,52% | 92,13% |
| FER2013 | 0,249672 | 91,24% | 87,35% | 50,02% | 49,67% |
| GTSRB | 0,000004 | 100,00% | 100,00% | 99,76% | 99,76% |
| KMNIST | 0,000000 | 100,00% | 100,00% | 98,81% | 98,77% |
| MNIST | 0,000000 | 100,00% | 100,00% | 99,28% | 99,26% |
| SVHN | 0,001428 | 99,99% | 98,72% | 92,54% | 92,33% |

O treino aproxima-se de 100% em vários datasets, enquanto CIFAR-10, CIFAR-100 coarse e FER2013 mantêm macro-F1 de teste substancialmente menor. As curvas também mostram que alguns melhores checkpoints surgem cedo. Isso é compatível com ajuste excessivo ou dificuldade de transferência dos embeddings, mas o relatório não atribui causalidade sem ablações.

| Dataset | Parede por execução | Média por época | Vazão média gravada | CPU média do processo | RSS máximo |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 26,0 s | 0.184 s | 230.909 ex/s | 110,9% | 1231,6 MB |
| CIFAR-100 coarse | 27,1 s | 0.185 s | 229.373 ex/s | 110,0% | 1243,7 MB |
| EMNIST Balanced | 62,8 s | 0.404 s | 228.842 ex/s | 117,3% | 1207,1 MB |
| Fashion-MNIST | 29,0 s | 0.206 s | 239.578 ex/s | 112,7% | 1077,5 MB |
| FER2013 | 15,6 s | 0.108 s | 236.600 ex/s | 106,3% | 1265,5 MB |
| GTSRB | 20,0 s | 0.123 s | 227.288 ex/s | 105,0% | 1263,8 MB |
| KMNIST | 29,9 s | 0.213 s | 233.231 ex/s | 111,0% | 1077,8 MB |
| MNIST | 32,5 s | 0.221 s | 225.545 ex/s | 108,9% | 1051,7 MB |
| SVHN | 41,9 s | 0.304 s | 230.330 ex/s | 116,7% | 1252,8 MB |

No conjunto dos nove runs: 900 épocas, aproximadamente 194,9 s somados em épocas e 281,6 s de ajuste registrados. A soma de wall_seconds_current_attempt é 284,7 s; esses valores são somas por dataset, não duração da extração de features.
A telemetria dos treinos Dense contém 124 amostras. Entre os runs, a CPU média do processo variou de 105,0% a 117,3%, com picos de 127,6% a 135,1%; o RSS observado variou de 934,0 a 1265,5 MB; a memória do sistema variou de 69,3% a 72,9%. gpu_utilization_percent ficou sem medição (null) nos sumários.

## Ambiente do treinamento Dense no Mac

- Plataforma reportada: macOS-15.5-arm64-arm-64bit; Python 3.12.3; TensorFlow 2.18.1.
- Dispositivos físicos vistos pelo TensorFlow no preflight: CPU.
- Espaço livre no preflight: 225,35 GiB.
- O preflight ocorreu em 29/09/2026 22:36:53 BRT e registrou dtype_policy: float32.

Os vetores de entrada já estavam pré-computados em WSL2/Linux antes do treino Dense no Mac. Os campos de plataforma nos manifests e nos registros de hardware da extração identificam essa etapa separada; não atribuímos a extração ao Mac.

## Inventário dos artefatos brutos

A cópia versionada das evidências compactas dos nove treinos Dense está em [reports/mac-training-data-2026-09-29-to-2026-09-30](reports/mac-training-data-2026-09-29-to-2026-09-30/README.md). Ela inclui históricos completos, avaliação final por classe, estatísticas de features, timing, status e telemetria resumida. Pesos, previsões/logits, matrizes e amostras individuais de telemetria permanecem nos outputs locais do Mac. As amostras individuais de telemetria da extração WSL2 também permanecem locais; a versão publicada conserva apenas os registros compactos ligados aos manifests finais.

Cada diretório outputs/classical-128-20-mac/<dataset>/seed-42/ contém 19 arquivos:

- history.csv: 100 linhas de época; loss/acurácia de treino, métricas de validação, duração, vazão e precision/recall/F1/suporte por classe em cada época.
- test_metrics.json e classification_report.json: avaliação final e métricas por classe; confusion_matrix.npy: matriz de confusão.
- predictions.csv e test_logits.npy: previsão e logits por exemplo do teste.
- config.json, manifest.json, status.json, timing.json, feature_statistics.json, model_summary.txt.
- best.keras, final.keras, backup/latest.weights.h5 e metadados do backup.
- telemetry/hardware.json, telemetry/samples.jsonl e telemetry/summary.json.

Os caches-fonte ficam em outputs/classic-dense20/<dataset>/runs/<run>/artifacts/features128/; cada cache tem os seis arrays de features/rótulos, manifest.json, status.json e telemetria. O array .npy preserva os valores completos, enquanto o manifesto registra forma, dtype e hashes.

| Dataset | Histórico completo | Métricas teste | Relatório por classe | Manifesto/hash das features | Estatísticas das features | Telemetria resumida |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar10/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar10/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar10/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/cifar10/runs/cifar10__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar10/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar10/seed-42/telemetry/summary.json) |
| CIFAR-100 coarse | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar100_coarse/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar100_coarse/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar100_coarse/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/cifar100_coarse/runs/cifar100_coarse__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar100_coarse/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/cifar100_coarse/seed-42/telemetry/summary.json) |
| EMNIST Balanced | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/emnist_balanced/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/emnist_balanced/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/emnist_balanced/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/emnist_balanced/runs/emnist_balanced__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/emnist_balanced/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/emnist_balanced/seed-42/telemetry/summary.json) |
| Fashion-MNIST | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/fashion_mnist/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fashion_mnist/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fashion_mnist/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/fashion_mnist/runs/fashion_mnist__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fashion_mnist/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fashion_mnist/seed-42/telemetry/summary.json) |
| FER2013 | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/fer2013/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fer2013/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fer2013/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/fer2013/runs/fer2013__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fer2013/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/fer2013/seed-42/telemetry/summary.json) |
| GTSRB | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/gtsrb/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/gtsrb/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/gtsrb/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/gtsrb/runs/gtsrb__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/gtsrb/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/gtsrb/seed-42/telemetry/summary.json) |
| KMNIST | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/kmnist/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/kmnist/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/kmnist/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/kmnist/runs/kmnist__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/kmnist/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/kmnist/seed-42/telemetry/summary.json) |
| MNIST | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/mnist/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/mnist/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/mnist/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/mnist/runs/mnist__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/mnist/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/mnist/seed-42/telemetry/summary.json) |
| SVHN | [history.csv](reports/mac-training-data-2026-09-29-to-2026-09-30/svhn/seed-42/history.csv) | [test_metrics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/svhn/seed-42/test_metrics.json) | [classification_report.json](reports/mac-training-data-2026-09-29-to-2026-09-30/svhn/seed-42/classification_report.json) | [manifest.json](outputs/classic-dense20/svhn/runs/svhn__dense20_fp32_aug05__seed-42/artifacts/features128/manifest.json) | [feature_statistics.json](reports/mac-training-data-2026-09-29-to-2026-09-30/svhn/seed-42/feature_statistics.json) | [summary.json](reports/mac-training-data-2026-09-29-to-2026-09-30/svhn/seed-42/telemetry/summary.json) |

Os agregados locais e o relatório anterior permanecem no diretório de outputs ignorado pelo Git; o pacote versionado de evidências listado acima é a cópia portátil usada por este relatório.

## Limites de interpretação

- A média entre nove tarefas diferentes é apenas descritiva; não foi calculada acurácia pooled.
- Há uma execução por dataset e somente seed 42; não existem repetições para intervalo de confiança ou variabilidade entre seeds.
- Os resultados medem as cabeças sobre embeddings congelados. Não demonstram melhora da CNN, nem vantagem de uma arquitetura quântica.
- Os registros deste diretório confirmam que os encoders foram preservados durante a extração. Eles não bastam para afirmar independência de todos os splits em relação ao treinamento original de cada encoder; a sobreposição precisa ser auditada dataset a dataset.

## Fontes de evidência

- reports/mac-training-data-2026-09-29-to-2026-09-30/preflight.json e os nove subdiretórios reports/mac-training-data-2026-09-29-to-2026-09-30/<dataset>/seed-42/.
- Nove caches em outputs/classic-dense20/<dataset>/runs/<run>/artifacts/features128/.
- Configuração da ponta em src/classic_models/vector_dense20.py.
