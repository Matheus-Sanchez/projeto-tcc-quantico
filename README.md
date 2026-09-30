# TCC — benchmark clássico de classificação de imagens

Este repositório contém somente o pipeline clássico e reproduzível do TCC. A
implementação experimental quântica e seus resultados foram removidos para
serem redesenhados em um trabalho separado.

## Estrutura

```text
src/
  data_prep/       carregamento, auditoria, registro e preparação dos dados
  classic_models/  backbone congelado, exportação features128 e cabeça densa
  metrics/         métricas e callbacks de avaliação
  logs/             telemetria de execução
  utils/            CLI e persistência atômica de artefatos
configs/            registros dos datasets locais
scripts/            atalhos para WSL e macOS
tests/              testes unitários do fluxo clássico
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
