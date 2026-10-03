# Cabeça TensorFlow com quatro PQCs locais

## Ambiente e protocolo

Python 3.12, TensorFlow 2.21.0 e NumPy 2.0.2 permanecem fixados. O extra
quantum fixa PennyLane 0.45.1, Qiskit 2.2.3 e Aer 0.17.2. Instalação a
partir da raiz:

~~~bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements/quantum-local.txt
python -m pip check
~~~

requirements/quantum-verified-lock.txt registra todas as versões do ambiente
Windows/Python 3.12 usado na verificação. Para reproduzi-lo, instale esse lock
antes de instalar o pacote com python -m pip install --no-deps -e .
O preflight exige as versões diretas e registra todas as transitivas.

utils.experiment_config.VectorHeadSettings centraliza seed 42, batch 128,
Adam, LR 0.0003, 100 épocas, FP32, seleção máxima por val_macro_f1, sem early
stopping ou scheduler, telemetria a cada 5 segundos. O carregador único em
data_prep.feature_vectors mantém os imports de classic_models.vector_dense20.
As features entram sem normalização adicional, PCA, seleção ou augmentation.
Manifests, hashes dos seis arquivos, labels e dimensões são validados antes
do primeiro job. O fingerprint de configuração impede retomar com outros dados.

Os simuladores são locais em CPU e precisão dupla: default.qubit e Aer
statevector; a interface Keras devolve FP32. Ruído usa default.mixed e Aer
density matrix. Não há provider, credenciais ou chamada a hardware remoto.
TensorFlow 2.21 em Windows nativo executa as densas em CPU; no WSL um dispositivo
GPU disponível pode executar a parte clássica. O ambiente efetivo é registrado.

## Circuito e gradientes

| Contrato | Valor |
|---|---|
| Entrada / saída Keras | [B,20] / [B,20] |
| Grupos | 0:5, 5:10, 10:15, 15:20 |
| Estado inicial | 00000 |
| Embedding | RX(x) uma vez, saída linear da Dense20 em radianos |
| Repetição | RZ(phi), RY(theta), RZ(omega) por wire, depois CNOTs |
| CNOTs | controle w, alvo (w+r)%5, controles 0 a 4, alcances 1,2,3 |
| Leitura | cinco expectativas Pauli Y por grupo |
| Pesos | [4,3,5,3], independentes, uniformes em [0,2π) |
| Total | 180 ângulos; 19.272 + 21*C parâmetros no modelo |
| Sete classes | 19.419 parâmetros treináveis |

Wire 0 é o bit à esquerda e o eixo mais significativo dos estados salvos.
Amplitudes e matrizes Aer são permutadas para essa convenção. Ler Y equivale
a aplicar S† e depois H antes de medir Z. RX(x)|0> tem expectativa Y -sin(x).

ParallelPQC20 é registrada para serialização Keras. tf.custom_gradient
envolve callbacks NumPy com gradientes explícitos das entradas e dos pesos.
O gradiente de um parâmetro é (f(θ+π/2)-f(θ-π/2))/2; cada avaliação retorna os
cinco observáveis juntos. O VJP contrai com o gradiente vindo da saída; os
gradientes dos pesos são somados sobre o batch. A média da loss já está no
gradiente recebido.

Há quatro processos persistentes, um por grupo, com dispositivo privado.
--simulation-batch limita as avaliações em cada chamada ao simulador sem
alterar o batch de otimização. O cálculo direto custa
4 * (1 + 2*(5+45)) = 404 avaliações lógicas de circuito por exemplo.
O perfil mede forward + backward de 128 exemplos sem atualizar o otimizador.
A estimativa exclui validação, sondas, persistência e estudo posterior.

Referências: [artigo](https://arxiv.org/pdf/2304.09224),
[tf.custom_gradient](https://www.tensorflow.org/api_docs/python/tf/custom_gradient).
A topologia segue a sequência explícita dos arquivos fornecidos.

## Comandos e saídas

~~~bash
# Verifica os nove caches e configura a matriz, sem simular nem treinar.
python -m quantum_models.parallel_dense20 --all --backend both --dry-run

# Teste de integração: 128 exemplos reais no treino, uma época por job.
python -m quantum_models.parallel_dense20 --all --backend both --smoke --resume

# Matriz completa: um job ativo; PennyLane e depois Qiskit para cada dataset.
python -m quantum_models.parallel_dense20 --all --backend both --resume

# Dataset e backend individuais.
python -m quantum_models.parallel_dense20 --dataset fer2013 --backend pennylane --resume

# Reavalia o checkpoint selecionado e completa os diagnósticos pendentes.
python -m quantum_models.parallel_dense20 --dataset fer2013 --backend qiskit --diagnostics-only

# Inspeção somente leitura; dispensa TensorFlow e workers.
python -m quantum_models.inspect --run outputs/quantum-parallel-128-20/runs/fer2013/qiskit/seed-42
~~~

Opções adicionais: --features-root, --output-root, --classical-root e
--simulation-batch (padrão 256). --all e --dataset são exclusivos.
O smoke usa um batch real de até 128, sondas ideais de duas amostras por classe,
validação/teste estratificados reduzidos e as 64 condições do estudo em oito
exemplos de teste e quatro sondas. Essas métricas são identificadas como
smoke_subset e não são comparadas com os resultados clássicos completos.

Saída padrão completa: outputs/quantum-parallel-128-20/runs/.
Smoke: outputs/quantum-parallel-128-20/smoke/.
Cada run está em <dataset>/<backend>/seed-42/.

~~~text
matrix.json / preflight.json / results.csv        na raiz da matriz
<run>/
  config.json / environment.json / feature_statistics.json
  initial_weights.npz / initialization.json / circuit.txt / circuit.json
  profile.json / model_summary.txt / history.csv / status.json / result.json
  checkpoints/state.json / epoch-NNNNN.keras / best.keras
  test_metrics.json / test_predictions.npz / classical_reference.json
  telemetry/<sessão>/hardware.json / samples.jsonl / summary.json
  diagnostics/
    batches/epoch-NNNNN/{train,validation}/batch-NNNNN.json
    batches.csv
    probes/{epoch-00000,epoch-NNNNN,best}.{npz,json}
    noise-study/selection.json
    noise-study/<geração>/conditions.csv / summary.json
    noise-study/<geração>/{counts,predictions,probes}/*.npz
~~~

As densas são inicializadas pelo mesmo procedimento Keras do modelo clássico.
O inicializador quântico usa um gerador NumPy separado, preservando essa
sequência. Os dois backends recebem pesos iniciais idênticos e permutações por
época de SeedSequence([42, época_zero_based]). Os hashes iniciais ficam salvos.
Referências históricas só são elegíveis com hashes, splits e protocolo
coincidentes. Tempo mantém o hardware e ambiente identificados: o treino
histórico usou Python 3.10.12, enquanto esta execução exige Python 3.12.

## Retomada e persistência

Um checkpoint .keras contém todas as variáveis e o estado do Adam. Primeiro
é salvo um arquivo temporário e substituído atomicamente; depois state.json
confirma seu hash, a época, a ordem dos exemplos e a melhor geração. Somente
essa confirmação é usada ao retomar. O checkpoint de melhor F1 e o último
são preservados. best.keras é uma cópia publicada da geração selecionada.

A retomada ocorre na última **época completa confirmada**. Uma interrupção
dentro da época refaz essa época desde o checkpoint anterior, com o mesmo
batch order e Adam. Arquivos de batches e sondas têm IDs determinísticos e
são substituídos; o CSV só incorpora épocas confirmadas. O histórico de
validação é truncado à época confirmada e substitui linhas da época reexecutada.
Sessões de telemetria têm UUID próprio. Uma run completa com --resume é
verificada e pulada. A matriz possui lock de processo com recuperação de
lock pertencente a um processo encerrado.

## Coleta e definições

Em cada batch de treino/validação e circuito, são registrados ângulos de
entrada, expectativas Y, norma/variância dos gradientes, atualização dos
ângulos, forward/backward, avaliações lógicas e throughput. A telemetria
compartilhada coleta sistema, processo principal, quatro workers, disco e
dados NVIDIA disponíveis. As métricas por época usam o callback compartilhado
de Macro-F1; nenhuma métrica de teste participa da seleção.

As duas sondas de validação por classe são fixas durante a run. Antes do treino,
ao final de cada época e no checkpoint selecionado, salvam-se estados após
embedding e cada repetição, probabilidades Z, Bloch XYZ, variância Y,
correlações/covariâncias XYZ de pares, densidades reduzidas de um/dois qubits,
purezas e entropias. Complexos ficam em NPZ, metadados em JSON, séries em CSV.
Estados completos permitem reconstruir outros observáveis posteriormente.

O estudo posterior mantém as densas e PQCs do checkpoint selecionado:
expectativas exatas e shots 256, 1.024, 4.096; escalas 0, 0,5, 1, 2;
cinco seeds 42 a 46 por condição com shots, total de 64 condições.
Após RX e cada tripla completa RZ-RY-RZ aplica-se despolarização p=0,002,
amplitude damping=0,001 e phase damping=0,001. Depois de cada CNOT, aplicam-se
os canais em controle e alvo, com despolarização p=0,01 por wire. Todas as
probabilidades são multiplicadas pela escala.

As matrizes de Kraus são compartilhadas: despolarização
(1-p)ρ + p/3*(XρX+YρY+ZρZ). Isso evita diferenças de parametrização entre SDKs.
Counts XYZ são amostrados independentemente das probabilidades das matrizes
simuladas em cada backend; bitstrings vão de 00000 a 11111, com q0 à esquerda.
Os cinco observáveis da mesma base usam a mesma amostra conjunta.
Erro padrão por observável: sqrt((1-<P>²)/shots).

Fidelity é a fidelidade Uhlmann ao quadrado, e trace distance é metade da
norma de traço da diferença entre densidades. Ambas comparam estados antes
da medição; shots alteram estimativas de observáveis e classificação.
Pureza global é Tr(ρ²). Entropia global/subsistemas é von Neumann em bits;
em estados ideais puros, a global é zero e a de um subsistema mede
emaranhamento. Entropia de probabilidades Z é Shannon, também em bits.
SNR auxiliar é 10 log10(mean(ideal_Y²)/mean((observed_Y-ideal_Y)²)); erro
zero registra null e zero_error_infinite.
MAE/RMSE das saídas e degradação de accuracy/Macro-F1 usam o teste após seleção.

## Verificação

~~~bash
python -m pytest -q
python -m quantum_models.parallel_dense20 --all --backend both --smoke --resume
~~~

Testes incluem topologia/sinal Y, equivalência de saídas/estados e Kraus,
parameter-shift contra diferenças finitas, Adam atualizando densas e cada ramo,
restauração de modelo/Adam em outro processo, interrupção antes/depois do commit,
deduplicação, e propriedades de densidades. Tolerâncias: 1e-9 em precisão dupla
e 1e-6 em FP32. Smokes verificam os nove caches reais com ambos os backends;
não equivalem aos treinos de 100 épocas.
