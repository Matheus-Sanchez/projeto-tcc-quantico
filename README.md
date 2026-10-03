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

Use Python 3.12 e crie um ambiente virtual na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\Activate.ps1         # PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No WSL com NVIDIA, use `requirements/classic-wsl-gpu.txt` e execute os
comandos por `scripts/wsl-classic-env.sh`. No macOS, use o perfil
`requirements/classical-128-20-mac.txt` para treinar a cabeça sobre vetores já
exportados.

## Uso

Os caminhos dos dados são definidos em `configs/datasets.yaml`; nenhum comando
baixa datasets.

```bash
# conferir configuração e integridade dos dados locais
tcc-benchmark datasets
tcc-benchmark audit --all --labels-only

# validar o pipeline sem treinar
bash scripts/run-dense20-all.sh --dry-run
bash scripts/run-features128-all.sh --verify-only

# treinar a cabeça vetorial a partir de features128 existentes
bash scripts/run-classical-128-20-mac.sh --all --dry-run
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
python -m pip install -r requirements/quantum-local.txt
python -m quantum_models.parallel_dense20 --all --backend both --dry-run
python -m quantum_models.parallel_dense20 --all --backend both --smoke --resume
python -m quantum_models.parallel_dense20 --all --backend both --resume
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
