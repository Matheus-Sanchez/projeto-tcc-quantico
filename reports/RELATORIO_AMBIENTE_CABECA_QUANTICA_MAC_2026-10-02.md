# Verificação do ambiente local para a nova cabeça quântica

Data: 02/10/2026. Checkout: `projeto-tcc`, branch `2mac`, commit de base
`794c6da6`. Verificação feita no macOS 15.5, arm64, Apple M4.

## Resultado

O ambiente local foi preparado para executar a cabeça `ParallelPQC20` em CPU.
O novo ambiente isolado é `.venv-quantum` (Python 3.12.3). O atalho
`scripts/run_hybrid_macos.sh` agora usa esse ambiente por padrão. A verificação
de dependências, os 33 testes do repositório e o `dry-run` dos 18 jobs passaram.
Nenhum treino quântico completo ou smoke de um batch foi iniciado nesta
verificação; o `dry-run` não simula circuitos nem treina modelos.

## Evidências observadas

| Item | Resultado |
| --- | --- |
| `.venv` antigo | TensorFlow 2.18.1, PennyLane 0.44.1 e Qiskit 2.3.0; o preflight da nova cabeça recusou essas versões. |
| `.venv-quantum` novo | TensorFlow 2.21.0, NumPy 2.0.2, PennyLane 0.45.1, Qiskit 2.2.3 e Aer 0.17.2; `runtime_versions(quantum=True)` retornou `ok=True`. |
| Dependências | `uv pip check --python .venv-quantum/bin/python`: 60 pacotes compatíveis. |
| Vetores de entrada | Os nove datasets locais passaram pela validação dos seis arquivos, dimensões, classes, finitude, manifests e hashes. |
| Matriz | `./scripts/run_hybrid_macos.sh --all --backend both --dry-run --output-root "$TMPDIR/projeto-tcc-quantum-final-dry-run"`: 18 jobs locais sequenciais configurados. |
| Testes | Com limites de threads equivalentes aos do executor: 33 aprovados, 31 avisos numéricos emitidos pelos testes. |
| TensorFlow no novo ambiente | Somente `CPU:0` detectado; multiplicação de matrizes executada em `CPU:0`; `physical_gpus=[]`. |

Uma primeira execução dos testes sem limitar as threads resultou em 12 casos
aprovados e dois erros de worker Qiskit (`OMP Error #178`, memória compartilhada
indisponível). Repetidos com `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `TF_NUM_INTRAOP_THREADS=1` e
`TF_NUM_INTEROP_THREADS=1`, os dois casos passaram e a suíte completa terminou
com 33 aprovações. O módulo de execução define esses limites antes de importar
TensorFlow e os SDKs quânticos; para executar `pytest` diretamente neste
contexto, use as variáveis explícitas no comando abaixo.

## Execução local

```bash
# Execute a partir da raiz do checkout.
uv pip check --python .venv-quantum/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  .venv-quantum/bin/python -m pytest -q
./scripts/run_hybrid_macos.sh --all --backend both --dry-run
```

O próximo passo de integração previsto pelo projeto é
`./scripts/run_hybrid_macos.sh --all --backend both --smoke --resume`, que inicia
treinamentos e produz artefatos sob `outputs/quantum-parallel-128-20/smoke/`.
Depois de validar o smoke, o comando da matriz de 100 épocas é
`./scripts/run_hybrid_macos.sh --all --backend both --resume`.

## Limites desta verificação

- A instalação nova não tem `tensorflow-metal`; portanto, tanto as camadas
  TensorFlow quanto os quatro simuladores locais rodam em CPU neste ambiente.
  Isto satisfaz o contrato funcional documentado da nova cabeça, mas tempo de
  treino e aceleração Metal não foram medidos.
- O `dry-run` valida configuração e vetores, mas não comprova throughput,
  estabilidade em 100 épocas, persistência de checkpoints em uma execução real
  ou os diagnósticos do smoke.
- Os arquivos locais já não versionados antes desta verificação foram
  preservados. `.venv-quantum` e `outputs/` são ignorados pelo Git.
