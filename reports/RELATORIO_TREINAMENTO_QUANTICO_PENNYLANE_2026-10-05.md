# A matriz PennyLane terminou nos nove datasets

Relatório apurado em 05/10/2026, horário de Brasília, para a execução completa
da nova cabeça quântica local sobre `features128`. A última execução terminou
em 05/10/2026 às 11h52; a [matriz](../training-artifacts/quantum-parallel-128-20/model-results/matrix_status.json)
foi marcada como concluída às 11h53. O backend desta matriz é **PennyLane**;
Qiskit não integra estes nove jobs.

## Resumo

- **9/9 datasets concluídos, 900/900 épocas confirmadas.** Os nove históricos
  têm 100 épocas consecutivas e os checkpoints da época final e da melhor
  validação passaram na conferência de hash.
- A avaliação usou o **teste completo** de cada dataset: 95.389 exemplos ao
  todo. O macro-F1 de teste vai de **47,12%** em CIFAR-100 coarse a **99,59%**
  em GTSRB. Cada valor usa o checkpoint escolhido por macro-F1 de validação.
- Os diagnósticos também terminaram: **576 condições de ruído/shots**, 918
  arquivos de sondas por época/seleção e 44.248 amostras de telemetria. O
  tempo somado das nove sessões foi **61,64 horas**. TensorFlow e os quatro
  simuladores locais executaram em CPU.
- A comparação automática com as cabeças clássicas foi **bloqueada pelo
  `split_fingerprint` ausente nas configurações clássicas**. Os hashes dos
  seis arquivos de features coincidem em cada dataset, mas nenhum delta
  clássico–quântico é tratado aqui como comparação aprovada.

## Progresso de cada etapa

| Etapa | Progresso verificado | Dados gerados e conferência |
| --- | --- | --- |
| Preflight e entradas | 9/9 | Python 3.12.3, TensorFlow 2.21.0, PennyLane 0.45.1; seis arrays por dataset, hashes e fingerprints das features atuais iguais aos registrados na run. |
| Inicialização e perfil | 9/9 | Configuração, pesos iniciais, sumário do modelo, circuito, estatísticas das features e perfil de um batch real. O perfil registrou 404 avaliações lógicas de circuito por exemplo de treino. |
| Treino e validação | 900/900 épocas | 900 linhas de histórico, 348.200 batches de treino e 75.300 de validação. Foram registrados 1.694.000 registros de batch por circuito, quatro por batch. |
| Checkpoints | 9/9 | `state.json` confirma a época 100 em todos os jobs; hashes dos checkpoints final e selecionado conferidos. `best.keras` está presente nos nove. |
| Seleção e teste | 9/9 | Melhor época escolhida por `val_macro_f1`; métricas, matriz de confusão, métricas por classe e predições para os 95.389 exemplos de teste. |
| Sondas e estudo posterior | 9/9 | 102 sondas por job: inicial, após cada época e checkpoint selecionado. Em cada job, 64 condições concluídas: ruído nas escalas 0, 0,5, 1 e 2; leitura exata ou 256, 1.024 e 4.096 shots com cinco seeds para shots finitos. |
| Telemetria e consolidação | 9/9 | Uma sessão de telemetria por job, 44.248 amostras no total; [9 linhas consolidadas](../training-artifacts/quantum-parallel-128-20/model-results/results.csv) e `matrix_status.json` com `completed`. |

As 348.200 passagens de batch de treino cobrem 44.523.100 exposições a
exemplos de treino ao longo das 100 épocas; são reutilizações dos mesmos
445.231 vetores de treino, não novos exemplos independentes. Os 1.694.000
registros incluem treino e validação, multiplicados pelos quatro circuitos.

## Resultado por dataset

Todos os jobs chegaram a **100 épocas**. As porcentagens estão arredondadas a
duas casas; os valores exatos permanecem nos arquivos `result.json` e
`test_metrics.json` de cada run.

| Dataset | Treino / validação / teste | Melhor época | Macro-F1 val. | Acurácia teste | Macro-F1 teste | Sessão | Fim BRT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [MNIST](../training-artifacts/quantum-parallel-128-20/model-results/mnist/result.json) | 48.999 / 10.503 / 10.498 | 13 | 99,37% | 99,30% | 99,30% | 6,75 h | 03/10 04h59 |
| [Fashion-MNIST](../training-artifacts/quantum-parallel-128-20/model-results/fashion_mnist/result.json) | 49.000 / 10.500 / 10.500 | 5 | 92,43% | 91,87% | 91,89% | 6,75 h | 03/10 11h43 |
| [KMNIST](../training-artifacts/quantum-parallel-128-20/model-results/kmnist/result.json) | 49.000 / 10.500 / 10.500 | 30 | 98,71% | 98,65% | 98,65% | 6,78 h | 03/10 18h30 |
| [EMNIST Balanced](../training-artifacts/quantum-parallel-128-20/model-results/emnist_balanced/result.json) | 92.120 / 19.740 / 19.740 | 40 | 88,85% | 88,52% | 88,47% | 12,76 h | 04/10 07h16 |
| [CIFAR-10](../training-artifacts/quantum-parallel-128-20/model-results/cifar10/result.json) | 42.000 / 9.000 / 9.000 | 12 | 73,35% | 73,26% | 73,32% | 5,84 h | 04/10 13h06 |
| [CIFAR-100 coarse](../training-artifacts/quantum-parallel-128-20/model-results/cifar100_coarse/result.json) | 42.000 / 9.000 / 9.000 | 23 | 48,43% | 47,06% | 47,12% | 5,85 h | 04/10 18h57 |
| [SVHN](../training-artifacts/quantum-parallel-128-20/model-results/svhn/result.json) | 69.502 / 14.894 / 14.893 | 9 | 92,59% | 93,02% | 92,63% | 9,62 h | 05/10 04h35 |
| [GTSRB](../training-artifacts/quantum-parallel-128-20/model-results/gtsrb/result.json) | 27.489 / 5.905 / 5.876 | 99 | 99,75% | 99,64% | 99,59% | 3,82 h | 05/10 08h24 |
| [FER2013](../training-artifacts/quantum-parallel-128-20/model-results/fer2013/result.json) | 25.121 / 5.384 / 5.382 | 96 | 50,23% | 50,85% | 48,52% | 3,47 h | 05/10 11h52 |

O checkpoint selecionado ocorreu cedo em sete datasets (épocas 5 a 40), mas
GTSRB e FER2013 selecionaram épocas 99 e 96. Isso **não encerrou** os treinos:
o protocolo executou as 100 épocas e preservou o melhor checkpoint de
validação. Os resultados entre datasets têm números de classes e dificuldades
diferentes; esta tabela não mede superioridade de uma arquitetura sobre outra.

### Sinais de generalização a investigar

Na época 100, a acurácia de treino/validação foi **99,32%/72,43%** em CIFAR-10,
**83,16%/47,39%** em CIFAR-100 coarse e **92,56%/51,34%** em FER2013. As
diferenças são observadas nos históricos e indicam generalização limitada
nesses três casos. O relatório não atribui uma causa: os dados disponíveis
não isolam efeitos da representação congelada, arquitetura ou otimização.

## Dados quânticos e de hardware coletados

O [estudo de ruído](../training-artifacts/quantum-parallel-128-20/model-results/mnist/noise-selection.json)
foi executado **depois** da escolha do checkpoint pelo conjunto de validação.
As 64 condições por job incluem um caso exato e 15 combinações de shots/seeds
para cada escala. As saídas registram acurácia, macro-F1, degradação relativa à
leitura ideal, erro das saídas, fidelidade de estado, distância de traço,
pureza, entropia e erro de amostragem. Há 576 arquivos de condições, 576
arquivos de predições e 540 arquivos de counts finitos no total. Esses
resultados são **simulação local**, não medições em hardware quântico.

Na escala de ruído **2**, com leitura exata, a fidelidade média de estado por
dataset ficou entre **0,5035 e 0,5161**. A variação da acurácia de
classificação em relação à escala 0 exata variou de **−0,211 ponto percentual**
(CIFAR-10) a **+0,130 ponto percentual** (FER2013). A mudança de estado e a
mudança de classificação medem coisas distintas; a estabilidade das decisões
nestas condições simuladas não comprova robustez em hardware real.

Cada run também guarda histórico por época com loss, acurácia, macro-F1 de
validação, precisão/recall/F1 e suporte por classe, tempo e throughput. O
teste guarda balanced accuracy, macro precision/recall/F1, AUC, matriz de
confusão e predições. As sondas guardam estados, expectativas, probabilidades
e medidas de emaranhamento; os diagnósticos de batch guardam entradas, saídas,
gradientes, atualizações e tempos por circuito. A telemetria inclui CPU,
memória, processos filhos, disco e dispositivo GPU quando disponível.

Foram coletadas **44.248 amostras de telemetria**. O maior RSS do processo
principal em cada job ficou entre **1,51 e 2,56 GB**; isso não é o pico de
memória da árvore completa de workers. O ambiente de cada run declara apenas
`CPU:0` para TensorFlow, quatro simuladores locais em CPU e nenhuma chamada
remota. Os arquivos completos das coletas e resultados portáteis estão em [`training-artifacts/quantum-parallel-128-20/`](../training-artifacts/quantum-parallel-128-20/README.md); os originais locais continuam em `outputs/`.

## Protocolo, fontes e limites da comparação

A matriz usou os vetores `features128` já exportados: não treinou novamente o
backbone nem aplicou nova normalização, PCA ou augmentation à cabeça. O
modelo é Dense(128, ReLU) → Dense(20, linear) → quatro PQCs locais de cinco
qubits → 20 expectativas Pauli Y → logits. O protocolo registra seed 42,
batch 128, Adam a 0,0003, 100 épocas, FP32 na interface Keras e simulação
quântica em precisão dupla. A seleção foi feita por máximo macro-F1 de
validação, sem early stopping ou scheduler. O teste só foi avaliado após a
seleção do checkpoint.

Fontes primárias portáteis: [matriz](../training-artifacts/quantum-parallel-128-20/model-results/matrix.json), [preflight](../training-artifacts/quantum-parallel-128-20/model-results/preflight.json), [resultados consolidados](../training-artifacts/quantum-parallel-128-20/model-results/results.csv) e, para cada dataset, `config.json`, `feature_statistics.json`, `history.csv`, `status.json`, `checkpoints/`, `test_metrics.json` e `classical_reference.json` em [`model-results/`](../training-artifacts/quantum-parallel-128-20/README.md). Os diretórios completos `diagnostics/` e `telemetry/` estão nos arquivos LFS descritos no README do pacote. Os números foram conferidos entre essas fontes; os hashes dos dois checkpoints referenciados por `state.json` e os seis arquivos de features de cada dataset também foram revalidados nesta apuração.

As nove referências clássicas estão marcadas como `eligible=false`, com
`differences=["split_fingerprint"]`: o campo falta no `config.json` clássico,
embora os hashes dos seis arquivos de features sejam idênticos aos usados
na matriz quântica. Isso é uma lacuna de metadados da comparação automática,
não evidência de que os vetores diferem. Não calculei deltas de acurácia ou
tempo contra a cabeça clássica. Para uma comparação formal, é preciso
reconciliar esse campo e conferir o protocolo e os ambientes efetivos.

O trabalho desta matriz PennyLane está **concluído**. Permanecem como análises
separadas a comparação formal com a cabeça clássica e qualquer execução
Qiskit; nenhuma delas é contabilizada como etapa pendente desta matriz.
