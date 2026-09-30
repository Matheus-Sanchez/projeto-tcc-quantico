# Benchmark do TCC

## Rodada clássica Dense20 com backbone congelado

Esta rodada é exclusivamente clássica. Ela seleciona o checkpoint FP32
controlado de cada dataset, congela o modelo até `block5_pool` e treina a
cabeça `GlobalAveragePooling2D -> Dense(20) -> LayerNormalization -> ReLU ->
Dense(C)`. O comando não importa nem executa `quantum_models`.

No Ubuntu WSL, crie um ambiente Python 3.10 isolado e exponha as bibliotecas
CUDA instaladas pelo TensorFlow:

```bash
python3.10 -m venv /root/.venvs/projeto-tcc-classic
/root/.venvs/projeto-tcc-classic/bin/python -m pip install -r requirements/classic-wsl-gpu.txt
export PYTHON_BIN=/root/.venvs/projeto-tcc-classic/bin/python
bash scripts/wsl-classic-env.sh "$PYTHON_BIN" -m classic_models.dense20 --all --dry-run
bash scripts/wsl-classic-env.sh "$PYTHON_BIN" -m classic_models.dense20 --all --registry configs/datasets.wsl.yaml
```

O perfil fixo usa seed 42, batch 128, FP32, Adam `3e-4`, augmentation 0,5
sem flip horizontal, 100 épocas e checkpoint por `val_macro_f1`. Use
`--resume` após interrupção. As saídas ficam em `outputs/classic-dense20/`.

### Interface correta para trocar a ponta densa

Para treinar uma nova ponta completa sem repetir o backbone, use **somente** a
exportação `features128`. Ela termina neste ponto:

```text
imagem -> backbone congelado -> block5_pool -> GlobalAveragePooling2D -> float32[128]
```

Não há `Dense(20)`, normalização ou ativação nesses vetores. `features20` e
`encoder20.keras` continuam sendo saídas específicas do experimento Dense20 e
não devem ser usados para comparar pontas densas completas diferentes.

Para validar ou refazer a exportação dos nove datasets no WSL:

```bash
export PYTHON_BIN=/root/.venvs/projeto-tcc-classic/bin/python
bash scripts/run-features128-all.sh --verify-only
bash scripts/run-features128-all.sh --overwrite
```

Cada run salva `artifacts/encoder128.keras` e os arquivos
`artifacts/features128/{train,val,test}_{features,labels}.npy`. O treino de uma
nova ponta deve carregar os vetores com `numpy.load`, construir o modelo com
`tf.keras.Input(shape=(128,))`, usar treino/validação para seleção e reservar o
split de teste para a avaliação final. A exportação também salva telemetria em
`artifacts/features128/telemetry/`.

### Ponta clássica 128 → 128(ReLU) → 20(tanh) → C

O runner `classic_models.vector_dense20` treina diretamente sobre os vetores
`features128`; ele não carrega imagens nem a CNN. A arquitetura fixa é:

```text
features128
→ Dense(128, activation="relu")
→ Dense(20)
→ tanh
→ Dense(C, activation=None)
→ logits
```

No macOS, após clonar o repositório, instale o Git LFS e prepare o ambiente:

```bash
git lfs install
git lfs pull
python3 -m venv .venv-classical-128-20
source .venv-classical-128-20/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/classical-128-20-mac.txt
```

Valide os nove conjuntos sem treinar:

```bash
PYTHON_BIN="$PWD/.venv-classical-128-20/bin/python" \
  bash scripts/run-classical-128-20-mac.sh --all --dry-run --fail-fast
```

Execute as nove rodadas de 100 épocas, com a barra de progresso do TensorFlow:

```bash
PYTHON_BIN="$PWD/.venv-classical-128-20/bin/python" \
  bash scripts/run-classical-128-20-mac.sh --all --max-epochs 100 --fail-fast
```

Os resultados ficam em `outputs/classical-128-20-mac/<dataset>/seed-42/`.
Use `--resume` no mesmo comando para retomar uma execução interrompida. Cada
run salva `best.keras`, `final.keras`, `history.csv`, métricas, predições,
matriz de confusão, tempos e amostras de telemetria de CPU, RAM, processo e
disco. A telemetria NVIDIA é registrada somente quando `nvidia-smi` existe.

O projeto tem quatro blocos pequenos e independentes:

- `data_prep`: leitura local, split estratificado, normalização e balanceamento;
- `classic_models`: CNN de referência em Keras;
- `quantum_models`: QCNN em PennyLane/Qiskit e integração com a CNN congelada;
- `metrics` e `utils`: relatórios, configuração e CLI.

## Ambiente

```powershell
python -m pip install -r requirements.txt
```

Os dados locais são definidos em `configs/datasets.yaml`. Nenhum comando baixa
datasets.

## Comando deste repositorio

`tcc-benchmark run --all --dry-run` pertence ao projeto anterior e nao executa
o experimento hibrido deste repositorio. Para rodar a arquitetura CNN congelada
-> QCNN -> Dense, use `python -m quantum_models.smoke` a partir da raiz deste
projeto.

## Smoke: CNN treinada e congelada → QCNN → Dense

O comando não treina uma nova CNN. Ele procura em `models/` um checkpoint
concluído equivalente ao `--dataset`, valida que ele usa `all_raw` e
`unit_interval`, expõe o vetor de 256 posições da camada
`global_pool_concat`, congela o extrator convolucional e treina somente o
circuito e a camada `Linear(256, 10)`.

```powershell
python -m quantum_models.smoke --dataset kmnist --backend pennylane --samples-per-class 12 --head-epochs 1
```

O smoke é uma validação de integração, não uma medição de acurácia final: usa
um subconjunto minúsculo e só uma época. Os resultados e checkpoints ficam em
`outputs/quantum-smoke/`.

### Teste de 50 epocas da cabeca quantica

```powershell
python -m quantum_models.smoke `
  --dataset kmnist `
  --backend pennylane `
  --model-root models `
  --samples-per-class 12 `
  --head-epochs 50 `
  --head-batch-size 2 `
  --head-train-per-class 1 `
  --head-validation-per-class 1 `
  --output-root outputs\quantum-model-backed-50ep
```

Esse comando usa o checkpoint já treinado de KMNIST, congela a sua parte
convolucional e treina apenas a QCNN e a camada densa por 50 épocas.

Por padrão, o seletor prefere `float32` e depois `relu`. No conjunto atual de
modelos, KMNIST não possui os dois atributos no mesmo checkpoint: o escolhido
é o FP32 com ativação Swish. A ativação pertence à cabeça clássica removida,
enquanto o vetor usado pelo circuito é anterior a ela. O resultado registra a
origem e o perfil real do checkpoint. Use `--strict-model-profile` somente
quando existir um checkpoint com o perfil exato desejado.

Para executar a mesma cabeça no Qiskit:

```powershell
python -m quantum_models.smoke --dataset kmnist --backend qiskit --samples-per-class 12 --head-epochs 1
```

O backend Qiskit usa simulação exata e gradiente reverso; por isso tende a ser
mais lento. Os dois backends compartilham topologia, inicialização e saída
multiclasse, e são testados quanto à equivalência numérica.

Os blocos QCNN adaptados de `takh04/QCNN` estão documentados em
`third_party/takh04-qcnn-NOTICE.md` e mantêm a licença Apache-2.0 correspondente.
