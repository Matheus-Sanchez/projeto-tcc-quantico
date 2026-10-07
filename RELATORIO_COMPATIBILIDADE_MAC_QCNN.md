# Relatório de compatibilidade — iMac M4 e QCNN híbrida

**Data:** 2026-09-28<br>
**Repositório:** `projeto-tcc-quantico`, commit `38283b0`
**Branch local deste Mac:** `codex/mac-imac-m4`

> **Atualização em 2026-09-28:** a indisponibilidade de GPU foi rastreada ao
> contexto de lançamento da sessão macOS, não às versões instaladas. O mesmo
> ambiente cria e usa `GPU:0` quando iniciado no domínio Aqua com
> `launchctl asuser`. O diagnóstico, o launcher seguro e as validações estão
> em [RELATORIO_DIAGNOSTICO_METAL_QCNN.md](RELATORIO_DIAGNOSTICO_METAL_QCNN.md).

## Veredito

O núcleo da arquitetura **CNN congelada → QCNN → Dense** funciona neste iMac M4 para os dois backends quânticos. As dependências foram instaladas em um ambiente isolado e o forward/backward dos circuitos PennyLane e Qiskit produziu gradientes finitos.

O experimento completo com dados reais deve ser iniciado pelo launcher
`./scripts/run_hybrid_macos.sh`, que entra no domínio Aqua da sessão gráfica e
exige um kernel TensorFlow em GPU antes da extração. A execução que originou
este relatório foi iniciada em outro contexto e, por isso, não deve ser usada
como medida de desempenho Metal.

O circuito quântico não depende da GPU Metal neste código: ele é explicitamente executado em CPU, em `float64`, tanto no `default.qubit` do PennyLane quanto no `Statevector` do Qiskit.

## Ambiente organizado e validado

| Item | Estado observado |
| --- | --- |
| Máquina | iMac `Mac16,3`, Apple M4 (10 núcleos de GPU), 16 GB de memória unificada |
| Sistema | macOS 15.5, ARM64; Metal suportado pelo hardware |
| Python do sistema | 3.9.6 — incompatível com o projeto |
| Ambiente criado | `.venv/` com CPython **3.12.3**, conforme `.python-version` |
| Dependências | 76 pacotes coerentes segundo `uv pip check` |
| Frameworks | TensorFlow 2.18.1, tensorflow-metal 1.2.0, PyTorch 2.8.0, PennyLane 0.44.1, Qiskit 2.3.0, Qiskit Aer 0.17.2 |
| Ferramentas locais | Xcode/clang ARM64 disponível; cerca de 230 GiB livres no volume de dados |

O `.venv/` é local e já é ignorado por Git. Foi instalado `pip` nele para que os comandos documentados em `ENVIRONMENT.md` também funcionem.

Os insumos da fase 1 foram importados em 2026-09-28 sem levar caches ou artefatos intermediários: os nove datasets ativos ocupam 1.22 GB em `datasets/`, e 18 modelos clássicos concluídos (nove FP32 e nove FP16) estão em `models/controlled-quantization-fast-mac-m4/quantization/`. Cada modelo foi copiado como o par mínimo `manifest.json` + `checkpoints/best.keras`; todos passaram na verificação SHA-256. Foram excluídos `.downloads/`, `.staging/`, o diretório aninhado `datasets/datasets/`, relatórios e checkpoints históricos não selecionados.

Também foi corrigida a reprodução do ambiente no repositório: `tensorflow-metal==1.2.0` agora aparece com marcador `darwin`/`arm64` em `requirements.txt` e no extra `tensorflow` de `pyproject.toml`. Antes desta revisão, o documento afirmava que o plug-in seria instalado automaticamente, mas nenhum manifesto o declarava.

O diretório `src/tcc_benchmark.egg-info/` contém metadados regeneráveis do pacote. A fonte de verdade das dependências e dos módulos incluídos é `pyproject.toml`; o `.gitignore` agora exclui essa saída gerada.

As decisões de ambiente seguem a documentação oficial: [Apple — TensorFlow Metal](https://developer.apple.com/metal/tensorflow-plugin/), [PyTorch — MPS](https://docs.pytorch.org/docs/main/notes/mps.html) e [PennyLane — instalação](https://docs.pennylane.ai/en/stable/development/guide/installation.html).

## O que já funciona

- `PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/projeto-tcc-pycache" .venv/bin/python -m compileall -q src tests` terminou com código 0.
- `.venv/bin/python -m pytest -q` passou: **6 passed in 4.17s**.
- `tcc-benchmark datasets` reconhece os nove datasets configurados e `python -m quantum_models.smoke --help` expõe a interface de execução.
- Um vetor de 256 características foi processado por cada backend; ambos retornaram logits `(1, 2)` e gradiente finito para os 51 parâmetros quânticos:
  - PennyLane: norma L2 do gradiente `0.001828830313203018`;
  - Qiskit: norma L2 do gradiente `0.0018288303132032305`.
- A equivalência de features dos dois backends é coberta pela suíte de testes.
- O audit local de KMNIST leu 70.000 exemplos, sem erros ou avisos.
- O seletor da fase 2 descobre os 18 modelos importados. O checkpoint FP32 do KMNIST carregou, expôs `global_pool_concat` com forma `(None, 256)` e permaneceu com zero variáveis treináveis.

Esses resultados confirmam a parte que é possível verificar sem executar a extração pela CNN: a topologia de 8 qubits, amplitude encoding de 256 valores, backpropagation do PennyLane e o gradiente reverso do Qiskit.

## Como a execução híbrida realmente funciona

O comando `python -m quantum_models.smoke` **não treina a CNN inicial**. Ele faz o seguinte:

1. procura um `manifest.json` concluído e um `checkpoints/best.keras` abaixo de `--model-root`;
2. exige `all_raw`, `unit_interval`, classes idênticas às do dataset e ausência de QAT;
3. carrega a CNN TensorFlow, obtém a camada `global_pool_concat` com 256 dimensões e a congela;
4. extrai vetores em TensorFlow;
5. treina somente os 51 parâmetros da QCNN e a camada `Linear(256, classes)` em PyTorch.

Isso está implementado em [smoke.py](src/quantum_models/smoke.py), especialmente na seleção do artefato, no congelamento do extrator e no loop da cabeça quântica. A função `normalize_features` em [qcnn.py](src/quantum_models/qcnn.py) rejeita tensores fora da CPU; portanto MPS não acelera a parte quântica atual. O modo Qiskit deve ser esperado como mais lento que PennyLane, pois usa simulação exata e `ReverseEstimatorGradient` por amostra.

## Bloqueios e ajustes necessários

### P0 — iniciar futuras rodadas no domínio Aqua e exigir preflight real

O hardware está saudável do ponto de vista de detecção: `system_profiler` e `ioreg` identificam Apple M4, 10 núcleos e Metal. Porém, nesta sessão os resultados foram:

```text
tf.config.list_physical_devices("GPU") -> []
torch.backends.mps.is_built() -> True
torch.backends.mps.is_available() -> False
```

Isso apontou para indisponibilidade de Metal no contexto que iniciou a execução, e não para falta de GPU ou de pacote. O diagnóstico posterior confirmou que o mesmo ambiente funciona no domínio Aqua. Para as próximas rodadas, use:

```bash
# Execute a partir da raiz do checkout.
./scripts/run_hybrid_macos.sh --help
```

Somente após esse comando concluir e informar um `matmul_device` em `GPU:0`
deve ser executada a extração pela CNN. O PyTorch MPS pode permanecer
indisponível sem impedir a QCNN, pois esta foi implementada para CPU.

### Resolvido — disponibilizar os insumos que o Git deliberadamente não contém

Os diretórios `datasets/`, `models/` e `outputs/` estão no `.gitignore`; os insumos necessários da fase 1 foram copiados para os dois primeiros. A procedência selecionada é a matriz concluída `controlled-quantization-fast-mac-m4/quantization`, com os dois perfis não-QAT para cada dataset. Para manter esse contrato nas próximas cópias, preserve:

- o dataset local escolhido, por exemplo `datasets/kmnist/`, ou um YAML passado em `--registry` que aponte para a cópia existente;
- um checkpoint CNN concluído em `models/.../checkpoints/best.keras` e seu `manifest.json` compatível;
- o manifesto com `status: "completed"`, `balance_mode: "all_raw"`, `normalization: "unit_interval"`, uma camada `global_pool_concat` de 256 posições e sem `qat_weight_bits`.

Não substitua esses artefatos por modelos fictícios: o validador usa essas condições para evitar um experimento híbrido inconsistente.

### P1 — manter a fronteira entre as fases 1 e 2 explícita

Há código que constrói e compila a CNN em `classic_models/legacy_cnn.py`, mas o único comando público, `tcc-benchmark`, faz listagem/auditoria de datasets. Nesta organização, isso é aceitável: a fase 1 produz a CNN clássica e a fase 2 a consome congelada. Mantenha, porém, a cópia sob `models/` com o manifesto original; assim o experimento registra qual CNN produziu as 256 características e não confunde resultados da fase 1 com as métricas quânticas.

### P2 — corrigir documentação histórica e explicitar a política de GPU

`RELATORIO_DATA_PREP.md` ainda referencia caminhos `file:///c:/...` e módulos que não correspondem ao layout atual em `src/data_prep/`. Ele não impede o programa, mas reduz a rastreabilidade. Além disso, o runner aceita a CNN em CPU quando TensorFlow não acha GPU; para preservar o protocolo neste Mac, convém adicionar uma opção de preflight/fail-fast que aborte a fase TensorFlow se não houver GPU.

## Primeiro teste real recomendado

Com o launcher Aqua, comece pelo PennyLane e por um smoke mínimo, com seed fixa:

```bash
# Execute a partir da raiz do checkout.
./scripts/run_hybrid_macos.sh \
  --dataset kmnist \
  --backend pennylane \
  --mode smoke \
  --samples-per-class 12 \
  --head-epochs 1 \
  --head-batch-size 2 \
  --head-train-per-class 1 \
  --head-validation-per-class 1 \
  --preferred-dtype float32 \
  --preferred-activation swish \
  --strict-model-profile \
  --seed 42
```

Use `--registry` e `--model-root` somente se os dados e modelos estiverem fora da raiz. O resultado deverá ir para `outputs/quantum-smoke/`; valide primeiro `run-status.json`, `experiment-result.json`, os relatórios de validação/teste e a telemetria. Execute Qiskit apenas depois do smoke PennyLane passar, mantendo a mesma seed e os mesmos insumos.

## Limites desta revisão

Na revisão inicial não havia treino real nem extração de features, pois o
contexto original não expunha a GPU TensorFlow. O diagnóstico posterior
validou um kernel Metal no domínio Aqua e adicionou proteção contra fallback.
O resultado de uma nova execução completa com essa proteção ainda deve ser
coletado separadamente.
