# Relatório completo da cabeça quântica PennyLane

Resultados da campanha `quantum_128_relu_20_parallel_pqc_features128`, seed 42, sobre os nove datasets locais. Corte de evidência: **05/10/2026 às 11h52 BRT**, término da última run. Apuração e figuras geradas em 05/10/2026. Originais preservados em `outputs/quantum-parallel-128-20/runs/<dataset>/pennylane/seed-42/`.

## Arquivos de execução

Os resultados por modelo e a coleta integral de diagnósticos/telemetria estão em [training-artifacts/quantum-parallel-128-20](../../training-artifacts/quantum-parallel-128-20/README.md).

## Resumo executivo

- **9/9 treinamentos, 900/900 épocas**, 445,231 vetores de treino e **95,389 exemplos de teste**. Macro-F1 vai de **47,12% em CIFAR-100 coarse a 99,59% em GTSRB**. Os resultados são do checkpoint escolhido por validação.
- As nove sessões somam **61,64 horas**. Foram lidos **44.248 registros de hardware**, **423.500 JSONs de batch** e **1.694.000 registros por circuito**. CPU da árvore média perto de 384%; RSS de pico da árvore entre **2,40 e 3,28 GiB**.
- Há **918 sondas ideais** e **576 condições de ruído/shots**; todos os campos dos tensores foram inventariados. O pacote contém **632 figuras científicas** em PNG e SVG editável. As figuras principais aparecem neste relatório; a galeria e o catálogo completos estão em um ZIP complementar.
- As nove comparações clássicas continuam inelegíveis: `split_fingerprint=null` nos configs clássicos, embora os seis hashes de features coincidam. **Não há delta clássico–quântico ou afirmação de vantagem quântica.**

## Como consultar todos os gráficos e dados

- A prévia interativa foi gerada localmente, mas não integra os arquivos publicados nesta branch.
- [Galeria completa de figuras científicas](figuras-completas.zip): contém as 632 figuras em PNG e SVG, o catálogo e a página HTML da galeria. Extraia o ZIP nesta pasta e abra `figures/index.html`.
- [Tabelas e verificações de evidência](evidence/): contêm históricos, amostras agregadas de hardware, resumos de batches e tensores, condições de ruído, counts e verificações. Os registros brutos continuam nos arquivos de execução em `outputs/`.

![Resultados dos nove datasets](figures/01-resultados.png)

## Resultados de treinamento e teste

| Dataset | Treino | Validação | Teste | Classes | Melhor época | Acurácia teste | Macro-F1 teste | Balanceada | AUC macro OVR | Sessão (h) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | 48999 | 10503 | 10498 | 10 | 13 | 99,30% | 99,30% | 99,30% | 0.999587 | 6,75 |
| Fashion-MNIST | 49000 | 10500 | 10500 | 10 | 5 | 91,87% | 91,89% | 91,87% | 0.988244 | 6,75 |
| KMNIST | 49000 | 10500 | 10500 | 10 | 30 | 98,65% | 98,65% | 98,65% | 0.999250 | 6,78 |
| EMNIST Balanced | 92120 | 19740 | 19740 | 47 | 40 | 88,52% | 88,47% | 88,52% | 0.995657 | 12,76 |
| CIFAR-10 | 42000 | 9000 | 9000 | 10 | 12 | 73,26% | 73,32% | 73,26% | 0.944588 | 5,84 |
| CIFAR-100 coarse | 42000 | 9000 | 9000 | 20 | 23 | 47,06% | 47,12% | 47,06% | 0.879735 | 5,85 |
| SVHN | 69502 | 14894 | 14893 | 10 | 9 | 93,02% | 92,63% | 92,56% | 0.992032 | 9,62 |
| GTSRB | 27489 | 5905 | 5876 | 43 | 99 | 99,64% | 99,59% | 99,69% | 0.999954 | 3,82 |
| FER2013 | 25121 | 5384 | 5382 | 7 | 96 | 50,85% | 48,52% | 47,52% | 0.800359 | 3,47 |

As 100 épocas foram executadas em todos os datasets, sem early stopping. Em sete datasets o melhor checkpoint apareceu entre as épocas 5 e 40; GTSRB e FER2013 escolheram 99 e 96. A época escolhida não encerra o treinamento. O teste não foi usado para seleção.

O número de classes e a dificuldade variam entre datasets. O resultado mais alto em um dataset não estabelece que a arquitetura tenha causado essa diferença. Há apenas uma seed de treinamento; não é possível estimar variância entre treinamentos independentes a partir destas runs.

### Generalização na época 100

| Dataset | Acurácia treino | Acurácia validação | Treino − validação (p.p.) |
| --- | --- | --- | --- |
| MNIST | 100,00% | 99,35% | 0,65 |
| Fashion-MNIST | 99,95% | 91,97% | 7,98 |
| KMNIST | 100,00% | 98,62% | 1,38 |
| EMNIST Balanced | 95,18% | 88,43% | 6,75 |
| CIFAR-10 | 99,32% | 72,43% | 26,89 |
| CIFAR-100 coarse | 83,16% | 47,39% | 35,77 |
| SVHN | 99,99% | 92,55% | 7,45 |
| GTSRB | 100,00% | 99,71% | 0,29 |
| FER2013 | 92,56% | 51,34% | 41,22 |

As maiores diferenças em CIFAR-10, CIFAR-100 coarse e FER2013 indicam generalização limitada nessas runs. Não há ablação que isole uma causa entre features congeladas, arquitetura e otimização. Isso é interpretação da diferença observada, não uma atribuição causal.

### Métricas de avaliação e por classe

O histórico preserva loss e acurácia de treino/validação, balanced accuracy e Macro-F1 de validação, LR, duração, exemplos e throughput, além de **F1, precisão, recall e suporte de cada classe em cada época**. O teste contém loss Keras, acurácia, balanced accuracy, Macro-F1, macro precisão/recall, AUC macro OVR, matriz de confusão e predições/logits/probabilidades. A diferença de precisão numérica entre Keras FP32 e as métricas de classificação não é tratada como resultado distinto.

Os heatmaps de classes cobrem todas as classes e todas as épocas. As matrizes de confusão mostram todas as células, inclusive zeros. Nomes de classe e suportes estão em `classes.csv`; os valores de teste correspondem ao melhor checkpoint, não à época 100.

## Hardware: sistema, principal, filhos e disco

![Tempo e hardware](figures/02-hardware-tempo.png)

| Dataset | Amostras | CPU sistema média (%) | CPU árvore média (%) | Pico RSS principal (GiB) | Pico RSS árvore (GiB) |
| --- | --- | --- | --- | --- | --- |
| MNIST | 4841 | 41,41 | 383,91 | 2,40 | 3,28 |
| Fashion-MNIST | 4843 | 41,66 | 384,37 | 1,97 | 3,00 |
| KMNIST | 4864 | 41,44 | 384,29 | 2,38 | 3,18 |
| EMNIST Balanced | 9159 | 42,22 | 384,28 | 2,50 | 3,24 |
| CIFAR-10 | 4191 | 42,37 | 384,54 | 1,97 | 2,96 |
| CIFAR-100 coarse | 4203 | 42,29 | 384,21 | 2,05 | 2,86 |
| SVHN | 6909 | 42,40 | 384,65 | 1,99 | 2,94 |
| GTSRB | 2744 | 42,73 | 384,14 | 1,74 | 2,56 |
| FER2013 | 2494 | 43,73 | 385,02 | 1,47 | 2,40 |

Os metadados observados registram macOS 15.5 ARM64, Python 3.12.3, 10 cores físicos/lógicos e RAM total de 16 GiB. TensorFlow registra apenas `CPU:0`; os simuladores usam CPU. Não há comprovação de operação GPU nestas runs.

### Definições e limitações da telemetria

- **CPU do sistema:** porcentagem total 0–100%. **CPU principal/filhos/árvore:** escala por core do psutil, podendo exceder 100%. A árvore é soma do principal e de todos os filhos observados; o coletor pode incluir processo auxiliar além dos quatro workers quânticos.
- **RSS e VMS:** campos `*_mb` foram calculados dividindo bytes por 1024², portanto são **MiB**, apesar do nome MB. Os gráficos em GiB dividem novamente por 1024. RSS da árvore soma memória dos processos e pode contar páginas compartilhadas.
- **RAM do sistema:** total, usada, disponível e percentual. Em memória unificada, não atribui consumo exclusivamente ao treino. Picos são máximos das amostras, não garantia de capturar picos entre intervalos.
- **Filhos:** cada amostra registra PID, nome, CPU, RSS e threads. PIDs não foram associados automaticamente a ramos específicos do circuito.
- **Disco:** total/usado/livre e percentual nos caminhos de outputs, features e run. Quando são o mesmo filesystem, não se somam esses valores. Variação do espaço livre não equivale à escrita exclusiva do treinamento.
- **Frequência:** preservada como campo informado pelo psutil; os valores não foram validados como frequência real do Apple Silicon.
- **Não coletados:** utilização GPU, potência, energia, temperatura, clocks validados, e I/O de processo quando indisponível. Ausência não é zero.
- **Fases:** as amostras de intervalo não mantêm `phase`; os eventos explícitos permitem localizar alguns marcos, mas não repartir integralmente as 61,64 horas entre treino, sondas e ruído.
- Médias são aritméticas entre amostras, não ponderadas por intervalo real. Os CSVs mantêm todas as amostras e timestamps UTC; o texto usa BRT.

## Circuitos, gradientes, parâmetros e custo

A arquitetura efetiva é **features128 → Dense(128, ReLU) → Dense(20, linear) → 4 PQCs de 5 qubits → 20 expectativas Pauli Y → logits**. Cada ramo recebe cinco ângulos crus, aplica RX uma vez, depois três repetições RZ–RY–RZ e cinco CNOTs por repetição. Alcances 1, 2 e 3, com q0 como bit à esquerda.

Por ramo: 45 ângulos treináveis, 65 portas elementares, 15 CNOTs, profundidade registrada 20. Total: **180 parâmetros quânticos**; o modelo completo tem 19.272 + 21 × classes parâmetros treináveis. Não é o QCNN histórico de 8 qubits e amplitude encoding.

Os gradientes usam parameter-shift ±π/2 nas entradas e nos pesos, com VJP explícito. Cada exemplo de treino registra **404 avaliações lógicas**, correspondentes a quatro ramos e aos deslocamentos. Na campanha completa, as contagens auditadas foram **17,987,332,400 avaliações de treino** e **38,170,400 de validação**. Essas contagens excluem perfis, sondas, teste e estudo posterior; não são shots nem execuções de QPU.

| Dataset | Parâmetros totais | Batches treino | Batches validação | Registros por circuito | Perfil (exemplos/s) | Estimativa de treino do perfil (h) |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 19482 | 38300 | 8300 | 186400 | 235,54 | 5,78 |
| Fashion-MNIST | 19482 | 38300 | 8300 | 186400 | 245,67 | 5,54 |
| KMNIST | 19482 | 38300 | 8300 | 186400 | 242,90 | 5,60 |
| EMNIST Balanced | 20259 | 72000 | 15500 | 350000 | 242,13 | 10,57 |
| CIFAR-10 | 19482 | 32900 | 7100 | 160000 | 240,00 | 4,86 |
| CIFAR-100 coarse | 19692 | 32900 | 7100 | 160000 | 242,57 | 4,81 |
| SVHN | 19482 | 54300 | 11700 | 264000 | 243,16 | 7,94 |
| GTSRB | 20175 | 21500 | 4700 | 104800 | 241,90 | 3,16 |
| FER2013 | 19419 | 19700 | 4300 | 96000 | 239,58 | 2,91 |

O perfil é um único batch real de 128 exemplos, forward/backward sem atualização do Adam. Sua extrapolação exclui validação, sondas, telemetria/persistência e estudo de ruído. Não substitui o tempo observado da campanha.

Cada JSON de batch guarda ângulos de entrada, expectativas Y, parâmetros, atualização dos parâmetros, gradientes de entrada/pesos, variância dos gradientes por exemplo, tempos forward/backward, avaliações lógicas, tempo físico, exemplos/s, avaliações/s e logs Keras. As séries resumem **todos** os batches por época/fase/circuito, com média, mínimo, máximo, desvio e contagem. Logs Keras de batch são cumulativos; sua média não é o resultado final da época. Média de variâncias de batch não é variância global de exemplos.

O tempo físico é somado uma vez por batch. Os tempos dos quatro circuitos são medidos em workers paralelos e se sobrepõem; somá-los não representa tempo de parede. Uma norma pequena de gradiente, sozinha, não comprova barren plateau.

## Estados e métricas quânticas

Há **102 snapshots ideais por dataset**: inicial (época 0), 100 épocas e a cópia do melhor checkpoint. A seleção de sondas mantém duas amostras fixas de validação por classe. Isso permite comparar a evolução das mesmas entradas, mas não representa toda a população. Cada snapshot contém quatro ramos e quatro estágios: embedding, layer_1, layer_2 e layer_3.

### Métricas no estágio final do checkpoint escolhido

| Dataset | Entropia 1 qubit (bits) | Entropia 2 qubits (bits) | Pureza 1 qubit | Pureza 2 qubits | Entropia Z (bits) | Correlação YY média |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 0,66 | 1,11 | 0,71 | 0,58 | 4,54 | -0,01 |
| cifar100_coarse | 0,73 | 1,23 | 0,67 | 0,53 | 4,47 | -0,04 |
| emnist_balanced | 0,48 | 0,78 | 0,80 | 0,71 | 4,64 | -0,03 |
| fashion_mnist | 0,66 | 1,13 | 0,71 | 0,58 | 4,57 | 0,03 |
| fer2013 | 0,81 | 1,39 | 0,62 | 0,46 | 4,54 | -0,02 |
| gtsrb | 0,30 | 0,47 | 0,88 | 0,83 | 4,76 | 0,03 |
| kmnist | 0,49 | 0,82 | 0,80 | 0,71 | 4,55 | 0,02 |
| mnist | 0,56 | 0,94 | 0,75 | 0,65 | 4,62 | 0,00 |
| svhn | 0,59 | 0,99 | 0,74 | 0,63 | 4,60 | -0,02 |

Valores acima são médias sobre os quatro ramos, sondas fixas e qubits/pares. Não são estimativas sobre todas as imagens. As séries temporais excluem `best` para não duplicar a época selecionada.

### Todos os arrays coletados

| Família | Conteúdo e interpretação |
|---|---|
| `indices`, `labels` | Identidade e classe das sondas; metadados, não variáveis contínuas de desempenho. |
| `angles`, `quantum_weights` | Ângulos de entrada (20 por sonda) e 180 parâmetros, em radianos. Valores crus, sem reduzir módulo 2π. |
| `statevectors` | 32 amplitudes complexas por ramo/estágio/sonda, ordem 00000…11111. |
| `density_matrices` | Estados mistos completos 32×32, usados nas sondas com ruído. |
| `probabilities_z` | Probabilidades dos 32 bitstrings na base Z. |
| `bloch_xyz` | Expectativas X/Y/Z dos cinco qubits. |
| `pauli_y_variance` | Variância de Y, 1 − ⟨Y⟩². |
| `yy_correlations`, `yy_covariance` | ⟨YᵢYⱼ⟩ e ⟨YᵢYⱼ⟩ − ⟨Yᵢ⟩⟨Yⱼ⟩ nos dez pares. |
| `two_qubit_pauli_correlations_xyz`, `two_qubit_pauli_covariance_xyz` | Matrizes 3×3 de correlações/covariâncias XYZ para cada par. |
| `reduced_one_qubit`, `reduced_two_qubit` | Densidades reduzidas 2×2 e 4×4. |
| `purity_global`, `purity_one_qubit`, `purity_two_qubit` | Tr(ρ²), global ou após traço parcial. |
| `entropy_global_bits`, `entropy_one_qubit_bits`, `entropy_two_qubit_bits` | Von Neumann −Tr(ρ log₂ρ), em bits. |
| `measurement_entropy_z_bits` | Shannon das probabilidades Z, em bits; não é entropia global de estado. |

Em simulação ideal, estado global puro tem pureza ≈1 e entropia global ≈0, com resíduos numéricos. Entropia reduzida pode refletir emaranhamento com o resto. Em estado misto com ruído, entropia reduzida inclui mistura e não é medida isolada de emaranhamento. A média de uma densidade sobre sondas é mistura estatística; não é uma densidade individual.

Todos os tensores foram lidos. O catálogo adicional contém 6.327 resumos de arrays, incluindo leituras XYZ e erros padrão das sondas, counts, logits, probabilidades e pesos iniciais. Os gráficos temporais usam estatísticas por ramo/estágio; o explorador de coordenadas mostra médias sobre as sondas no melhor checkpoint, separando partes real/imaginária/módulo. Os arrays complexos individuais completos permanecem nos NPZ originais. Cada coordenada e shape está documentada nos CSVs `probe_details.csv` e `tensor_catalog.csv`.

## Estudo posterior: ruído, shots e seeds

São quatro escalas de ruído (0, 0,5, 1, 2), leitura exata e três quantidades de shots (256, 1.024, 4.096). Para cada leitura finita, seeds 42–46: **4 × (1 + 3 × 5) = 64 condições por dataset**, 576 no total. Os canais são simulados em `default.mixed` após a seleção por validação. Não foi executado treinamento sob ruído nem hardware remoto.

Na escala 2 com leitura exata, fidelidade média fica entre **0.5035 e 0.5161**; a mudança de acurácia fica entre **-0,21 e 0,13 p.p.**. A fidelidade pode cair substancialmente sem mudança equivalente na decisão da classe; são métricas distintas.

| Dataset | Acurácia escala 2 exata | Mudança (p.p.) | Macro-F1 | Fidelidade média | Distância de traço média | Pureza global média | Entropia global (bits) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | 99,30% | -0,01 | 99,29% | 0,51 | 0,49 | 0,27 | 3,28 |
| fashion_mnist | 91,81% | -0,06 | 91,85% | 0,51 | 0,49 | 0,27 | 3,28 |
| kmnist | 98,65% | -0,00 | 98,65% | 0,52 | 0,48 | 0,28 | 3,24 |
| emnist_balanced | 88,50% | -0,02 | 88,47% | 0,51 | 0,49 | 0,27 | 3,28 |
| cifar10 | 73,04% | -0,21 | 73,07% | 0,51 | 0,49 | 0,27 | 3,29 |
| cifar100_coarse | 47,08% | 0,02 | 47,23% | 0,51 | 0,49 | 0,27 | 3,27 |
| svhn | 92,97% | -0,05 | 92,60% | 0,51 | 0,49 | 0,28 | 3,25 |
| gtsrb | 99,64% | -0,00 | 99,59% | 0,51 | 0,49 | 0,28 | 3,26 |
| fer2013 | 50,98% | 0,13 | 48,44% | 0,50 | 0,50 | 0,26 | 3,33 |

As figuras mostram todas as métricas registradas nas condições, comparando shots e escalas. Linha = média das cinco seeds; faixa = mínimo–máximo observado. **A faixa não é intervalo de confiança e as seeds não são cinco treinamentos.** O relatório interativo preserva as cinco curvas individuais por dataset.

### Definições de ruído e erro

- Canais após RX e tripla RZ–RY–RZ: despolarização p=0,002, amplitude damping=0,001 e phase damping=0,001. Após CNOT: despolarização p=0,01 por wire em controle e alvo. Cada probabilidade é multiplicada pela escala; os operadores de Kraus são os registrados no código.
- Fidelidade: Uhlmann **ao quadrado** entre ideal e ruidoso antes de medir. Distância de traço: 0,5 × norma de traço da diferença. Shots alteram leitura/classificação, sem alterar o estado antes da medição.
- MAE/RMSE e erro L2 relativo comparam expectativas Y observadas com as ideais no teste completo. Erro padrão médio de Y deriva de √((1−⟨Y⟩²)/shots).
- SNR auxiliar = 10 log₁₀(mean(ideal_Y²)/mean(erro²)), em dB. Caso ideal com erro zero: `snr_db_auxiliary=null` e `snr_status=zero_error_infinite`; representa **+∞**, não zero ou falta de coleta comum.
- Degradação registrada = baseline − observado; mudança em p.p. = −100 × degradação. Valores negativos de degradação indicam aumento observado, não erro de sinal.
- Counts X/Y/Z são multinomiais independentes derivados da densidade simulada. Frequência por bitstring = soma de counts/(exemplos × shots). Há **540 arquivos de counts finitos**, todos verificados. Os cinco observáveis da mesma base compartilham a amostra conjunta.

## Protocolo, origem das features e comparação clássica

Seed 42; Adam; LR 0,0003; batch 128; 100 épocas; FP32 na interface Keras; float64/complex128 no simulador; quatro workers persistentes; sem scheduler/early stopping. Features128 entram sem normalização adicional, PCA, seleção ou augmentation na cabeça. O nome `aug05` identifica a campanha fonte; não é augmentation aplicado aos vetores desta etapa.

Os manifests registram features de `block5_pool` após GlobalAveragePooling2D, nenhuma camada densa incluída, checkpoint/encoder/splits e hashes. O relatório considera a proveniência registrada. **Não foi refeita a auditoria de independência entre esses splits e o treinamento original do backbone**; por isso, as métricas não são apresentadas como validação externa de toda a cadeia.

Ambiente efetivo preservado: Python 3.12.3, TensorFlow 2.21.0, PennyLane 0.45.1, NumPy 2.0.2; o arquivo `metadata.json` contém todos os pacotes transitivos, dispositivos, perfis, configurações, inicializadores, sumários e circuito. A presença de Qiskit no ambiente não transforma estas runs em runs Qiskit.

A conferência atual confirma `split_fingerprint=null` nos nove configs clássicos e igualdade dos seis hashes de features entre clássico e quântico. Isso é lacuna de metadados, não prova de vetores diferentes. A comparação automática permanece bloqueada. Reconciliar o campo e revisar protocolo/ambiente é necessário antes de uma comparação formal; nenhum arquivo de treinamento foi alterado para contornar essa restrição.

## Conferências realizadas e cobertura

| Conferência | Resultado |
|---|---|
| Épocas consecutivas e runs completas | 9/9 completas; 900 linhas, épocas 1–100. |
| Seleção por validação | Época/melhor Macro-F1 conferidos com o histórico. |
| Teste e suportes | Status, métricas, suportes e soma da matriz de confusão conciliados. |
| Todos os batches | 423.500 JSONs lidos; 4 ramos por batch; 1.694.000 linhas conciliadas com CSV. |
| Avaliações lógicas | Treino = exemplos × 100 × 404; validação = exemplos × 100 × 4. |
| Sondas ideais | 918 verificações de normalização; zero falhas acima de 1e−9. |
| Counts finitos | 540 NPZs; contagens somam shots em todo exemplo/ramo/base. |
| Dados de gráficos | Derivados exclusivamente dos CSVs auditados, sem treinamento ou nova simulação. |
| Checkpoints | State/metadados lidos; modelos não carregados e hashes de checkpoints não foram recalculados nesta etapa. |
| Gráficos interativos | Build do runtime protegido aprovado; inspeção visual de navegador indisponível por política de URL. |
| Figuras estáticas | Arquivos PNG/SVG gerados; checagem de integridade e inspeção visual representativa registrada em verification.json. |

## Reprodução e organização dos arquivos

- `build_evidence.py`: lê a campanha completa, confere invariantes e produz CSVs/snapshot. Usa NumPy e pandas; não modifica as runs.
- `render_figures.py`: produz figuras científicas PNG/SVG com Matplotlib a partir dos CSVs auditados.
- `evidence/`: históricos, classes/confusão, features, amostras agregadas de hardware, resumos de batch e sondas, condições de ruído, counts, inventários e verificações. O `metadata.json` com caminhos absolutos desta máquina foi excluído.
- `figures/`: PNGs principais usados no relatório. O arquivo `figuras-completas.zip` contém as 632 figuras em PNG/SVG, o catálogo e a galeria HTML.
- A prévia interativa local e seu snapshot de aproximadamente 304 MiB não fazem parte do pacote publicado.

Os CSVs usam UTF-8 e quebras LF. IDs e datas não são compactados. Médias e dispersões são descritas no escopo de cada gráfico; nulos são preservados. O snapshot completo é grande porque retém todas as amostras e resumos para inspeção; a galeria de figuras pode ser consultada sem carregar esse snapshot.

## Figuras principais por dataset

### MNIST

![MNIST — aprendizado](figures/mnist-aprendizado.png)

![MNIST — loss](figures/mnist-loss.png)

![MNIST — cpu](figures/mnist-cpu.png)

![MNIST — rss](figures/mnist-rss.png)

![MNIST — entropy_one_qubit_bits_mean](figures/mnist-entropy_one_qubit_bits_mean.png)

![MNIST — purity_one_qubit_mean](figures/mnist-purity_one_qubit_mean.png)

![MNIST — ruido-accuracy_change_pp](figures/mnist-ruido-accuracy_change_pp.png)

![MNIST — ruido-fidelity_mean](figures/mnist-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de MNIST](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### Fashion-MNIST

![Fashion-MNIST — aprendizado](figures/fashion_mnist-aprendizado.png)

![Fashion-MNIST — loss](figures/fashion_mnist-loss.png)

![Fashion-MNIST — cpu](figures/fashion_mnist-cpu.png)

![Fashion-MNIST — rss](figures/fashion_mnist-rss.png)

![Fashion-MNIST — entropy_one_qubit_bits_mean](figures/fashion_mnist-entropy_one_qubit_bits_mean.png)

![Fashion-MNIST — purity_one_qubit_mean](figures/fashion_mnist-purity_one_qubit_mean.png)

![Fashion-MNIST — ruido-accuracy_change_pp](figures/fashion_mnist-ruido-accuracy_change_pp.png)

![Fashion-MNIST — ruido-fidelity_mean](figures/fashion_mnist-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de Fashion-MNIST](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### KMNIST

![KMNIST — aprendizado](figures/kmnist-aprendizado.png)

![KMNIST — loss](figures/kmnist-loss.png)

![KMNIST — cpu](figures/kmnist-cpu.png)

![KMNIST — rss](figures/kmnist-rss.png)

![KMNIST — entropy_one_qubit_bits_mean](figures/kmnist-entropy_one_qubit_bits_mean.png)

![KMNIST — purity_one_qubit_mean](figures/kmnist-purity_one_qubit_mean.png)

![KMNIST — ruido-accuracy_change_pp](figures/kmnist-ruido-accuracy_change_pp.png)

![KMNIST — ruido-fidelity_mean](figures/kmnist-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de KMNIST](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### EMNIST Balanced

![EMNIST Balanced — aprendizado](figures/emnist_balanced-aprendizado.png)

![EMNIST Balanced — loss](figures/emnist_balanced-loss.png)

![EMNIST Balanced — cpu](figures/emnist_balanced-cpu.png)

![EMNIST Balanced — rss](figures/emnist_balanced-rss.png)

![EMNIST Balanced — entropy_one_qubit_bits_mean](figures/emnist_balanced-entropy_one_qubit_bits_mean.png)

![EMNIST Balanced — purity_one_qubit_mean](figures/emnist_balanced-purity_one_qubit_mean.png)

![EMNIST Balanced — ruido-accuracy_change_pp](figures/emnist_balanced-ruido-accuracy_change_pp.png)

![EMNIST Balanced — ruido-fidelity_mean](figures/emnist_balanced-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de EMNIST Balanced](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### CIFAR-10

![CIFAR-10 — aprendizado](figures/cifar10-aprendizado.png)

![CIFAR-10 — loss](figures/cifar10-loss.png)

![CIFAR-10 — cpu](figures/cifar10-cpu.png)

![CIFAR-10 — rss](figures/cifar10-rss.png)

![CIFAR-10 — entropy_one_qubit_bits_mean](figures/cifar10-entropy_one_qubit_bits_mean.png)

![CIFAR-10 — purity_one_qubit_mean](figures/cifar10-purity_one_qubit_mean.png)

![CIFAR-10 — ruido-accuracy_change_pp](figures/cifar10-ruido-accuracy_change_pp.png)

![CIFAR-10 — ruido-fidelity_mean](figures/cifar10-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de CIFAR-10](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### CIFAR-100 coarse

![CIFAR-100 coarse — aprendizado](figures/cifar100_coarse-aprendizado.png)

![CIFAR-100 coarse — loss](figures/cifar100_coarse-loss.png)

![CIFAR-100 coarse — cpu](figures/cifar100_coarse-cpu.png)

![CIFAR-100 coarse — rss](figures/cifar100_coarse-rss.png)

![CIFAR-100 coarse — entropy_one_qubit_bits_mean](figures/cifar100_coarse-entropy_one_qubit_bits_mean.png)

![CIFAR-100 coarse — purity_one_qubit_mean](figures/cifar100_coarse-purity_one_qubit_mean.png)

![CIFAR-100 coarse — ruido-accuracy_change_pp](figures/cifar100_coarse-ruido-accuracy_change_pp.png)

![CIFAR-100 coarse — ruido-fidelity_mean](figures/cifar100_coarse-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de CIFAR-100 coarse](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### SVHN

![SVHN — aprendizado](figures/svhn-aprendizado.png)

![SVHN — loss](figures/svhn-loss.png)

![SVHN — cpu](figures/svhn-cpu.png)

![SVHN — rss](figures/svhn-rss.png)

![SVHN — entropy_one_qubit_bits_mean](figures/svhn-entropy_one_qubit_bits_mean.png)

![SVHN — purity_one_qubit_mean](figures/svhn-purity_one_qubit_mean.png)

![SVHN — ruido-accuracy_change_pp](figures/svhn-ruido-accuracy_change_pp.png)

![SVHN — ruido-fidelity_mean](figures/svhn-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de SVHN](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### GTSRB

![GTSRB — aprendizado](figures/gtsrb-aprendizado.png)

![GTSRB — loss](figures/gtsrb-loss.png)

![GTSRB — cpu](figures/gtsrb-cpu.png)

![GTSRB — rss](figures/gtsrb-rss.png)

![GTSRB — entropy_one_qubit_bits_mean](figures/gtsrb-entropy_one_qubit_bits_mean.png)

![GTSRB — purity_one_qubit_mean](figures/gtsrb-purity_one_qubit_mean.png)

![GTSRB — ruido-accuracy_change_pp](figures/gtsrb-ruido-accuracy_change_pp.png)

![GTSRB — ruido-fidelity_mean](figures/gtsrb-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de GTSRB](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)

### FER2013

![FER2013 — aprendizado](figures/fer2013-aprendizado.png)

![FER2013 — loss](figures/fer2013-loss.png)

![FER2013 — cpu](figures/fer2013-cpu.png)

![FER2013 — rss](figures/fer2013-rss.png)

![FER2013 — entropy_one_qubit_bits_mean](figures/fer2013-entropy_one_qubit_bits_mean.png)

![FER2013 — purity_one_qubit_mean](figures/fer2013-purity_one_qubit_mean.png)

![FER2013 — ruido-accuracy_change_pp](figures/fer2013-ruido-accuracy_change_pp.png)

![FER2013 — ruido-fidelity_mean](figures/fer2013-ruido-fidelity_mean.png)

[Galeria completa (ZIP) de FER2013](figuras-completas.zip) · [Históricos completos](evidence/history.csv) · [Condições de ruído](evidence/noise.csv)
