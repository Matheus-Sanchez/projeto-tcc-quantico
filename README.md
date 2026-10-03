# TCC — benchmark clássico e híbrido de classificação de imagens

Este repositório contém o pipeline clássico reproduzível e a cabeça híbrida
TensorFlow com quatro circuitos locais de cinco qubits. Ambos reutilizam
diretamente as mesmas features128 e os mesmos splits.

## Estrutura

```text
src/
  data_prep/       carregamento, auditoria, registro e preparação dos dados
  classic_models/  backbone congelado, exportação features128 e cabeça densa
  quantum_models/  PQCs, PennyLane/Qiskit locais, gradientes e diagnósticos
  metrics/         métricas e callbacks de avaliação
  logs/             telemetria de execução
  utils/            CLI e persistência atômica de artefatos
configs/            registros dos datasets locais
scripts/            atalhos para WSL e macOS
tests/              testes do fluxo clássico, circuitos, gradientes e retomada
```

O fluxo de treinamento é fixo e explícito:

```text
dados locais → split estratificado → backbone FP32 congelado
→ features128 → Dense(128, ReLU) → Dense(20, tanh) → logits
```

## Instalação

O runtime comum está fixado em Python 3.12.3 e os pacotes usam
`requirements/quantum-verified-lock.txt` como lock compartilhado pelo Mac e
pelo Windows/WSL. No macOS Apple Silicon, instale `uv` e prepare o ambiente:

```bash
brew install uv git-lfs
git lfs install --local
git lfs pull
bash scripts/setup-macos.sh
source .venv/bin/activate
```

O perfil `requirements/classical-128-20-mac.txt` instala o mesmo lock do
ambiente Windows/WSL. No Mac, TensorFlow e os simuladores quânticos executam em
CPU; o perfil `requirements/classic-wsl-gpu.txt` acrescenta CUDA no WSL. Use os
scripts `run-*-all.sh` somente dentro do WSL; eles apontam para caminhos e
ambiente Linux.

O relatório web usa Node.js 24.21.0, fixado em
`reports/dense20-classic-report/.nvmrc`. Com `nvm` instalado pelo Homebrew:

```bash
brew install nvm
mkdir -p "$HOME/.nvm"
export NVM_DIR="$HOME/.nvm"
source "$(brew --prefix nvm)/nvm.sh"
cd reports/dense20-classic-report
nvm install
npm ci
npm run dev -- --host 127.0.0.1
```

O Vite exibirá no terminal o endereço local do relatório.

## Uso

Os caminhos dos dados são definidos em `configs/datasets.yaml`; nenhum comando
baixa datasets.

```bash
# conferir configuração e integridade dos dados locais
tcc-benchmark datasets
tcc-benchmark audit --all --labels-only
```

Windows/WSL com NVIDIA:

```bash
bash scripts/run-dense20-all.sh --dry-run
bash scripts/run-features128-all.sh --verify-only
```

macOS com os vetores locais:

```bash
bash scripts/run-classical-128-20-mac.sh --all --dry-run
```

Para treinar a cabeça vetorial de verdade, remova `--dry-run`:

```bash
bash scripts/run-classical-128-20-mac.sh --all
```

Para executar o ensaio quântico reduzido nos dois simuladores locais:

```bash
python -m quantum_models.parallel_dense20 --all --backend both --smoke --resume
```

Os resultados clássicos são regeneráveis e permanecem em `outputs/`, fora do
versionamento para novas execuções.

## Evidências dos treinamentos no Mac

Os relatórios e históricos versionados dos nove treinamentos Dense feitos no
Mac em 29–30/09/2026 estão em
[`RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md`](RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md)
e em `reports/mac-training-data-2026-09-29-to-2026-09-30/`. Esses resultados
usaram features extraídas antes no WSL2/Linux; o relatório descreve os
ambientes e métricas históricos sem misturá-los aos treinos da nova cabeça.

## Cabeça quântica local

```bash
.venv/bin/python -m quantum_models.parallel_dense20 --all --backend both --dry-run
.venv/bin/python -m quantum_models.parallel_dense20 --all --backend both --smoke --resume
.venv/bin/python -m quantum_models.parallel_dense20 --all --backend both --resume
```

O treino completo executa 18 jobs, um por vez, na ordem dos nove datasets:
PennyLane e depois Qiskit. Cada job usa quatro workers persistentes. `--smoke`
usa uma época e um batch real, com avaliação reduzida; sua saída fica separada
da matriz completa. O teste de custo usa um batch real antes do treinamento.

O fluxo híbrido substitui somente a ativação tanh:

```text
features128 → Dense(128, ReLU) → Dense(20, linear)
→ 4 × PQC(5 qubits, 3 repetições) → 20 expectativas Y → Dense(C, linear)
```

As três densas e os 180 ângulos quânticos começam do zero. Não se carregam
pesos das cabeças clássicas. Treinamento, retomada, coleta e definições dos
diagnósticos estão descritos em [docs/quantum-parallel.md](docs/quantum-parallel.md).
