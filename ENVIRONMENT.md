# Ambiente dos experimentos

## Ambiente atual

O fluxo refatorado usa Python 3.12, NumPy 2.0.2 e TensorFlow 2.21.0. As
versões dos simuladores locais PennyLane e Qiskit estão fixadas em
`requirements/quantum-local.txt`; o lock verificado para Windows está em
`requirements/quantum-verified-lock.txt`.

No macOS, crie um ambiente virtual a partir da raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/classical-128-20-mac.txt
python -m pip install -r requirements/quantum-local.txt
python -m pip check
```

O perfil macOS usa TensorFlow em CPU. Os circuitos quânticos são simulados
localmente em CPU e precisão dupla; não requerem credenciais nem acesso a
hardware quântico remoto. Para executar a nova cabeça em um dataset:

```bash
./scripts/run_hybrid_macos.sh --dataset kmnist --backend both --dry-run
./scripts/run_hybrid_macos.sh --dataset kmnist --backend both --smoke --resume
```

A matriz completa e a retomada de execuções estão descritas em
[`docs/quantum-parallel.md`](docs/quantum-parallel.md).

## Evidências históricas do Mac

Os nove treinamentos Dense registrados em
`RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md` foram executados com
Python 3.12.3 e TensorFlow 2.18.1. O preflight desses treinamentos viu somente
CPU. Os vetores `features128` usados como entrada haviam sido extraídos em
WSL2/Linux; a execução no Mac treinou e avaliou as cabeças densas.

`src/quantum_models/metal_preflight.py` continua disponível para validar uma
GPU TensorFlow antes de uma extração de CNN feita no Mac. Essa verificação é
separada do treinamento da cabeça quântica atual, que consome vetores
pré-computados e funciona no perfil CPU.
