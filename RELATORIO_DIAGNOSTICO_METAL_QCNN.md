# Diagnóstico do TensorFlow Metal nas rodadas híbridas

**Data da investigação:** 2026-09-28<br>
**Máquina:** iMac Apple M4, macOS 15.5 (24F74), 16 GB de memória unificada
**Escopo:** explicar por que o run híbrido ativo não usou Metal e prevenir o mesmo fallback nas próximas rodadas. O treinamento ativo não foi interrompido.

## Conclusão

O problema **não é falta de GPU, incompatibilidade da arquitetura ARM64 nem instalação ausente do `tensorflow-metal`**. A causa observada é o **contexto de lançamento do processo**: o terminal integrado do VS Code que iniciou a execução atual não conseguiu inicializar uma GPU TensorFlow utilizável. Ao reexecutar o mesmo Python, as mesmas versões e o mesmo plug-in no domínio gráfico Aqua do usuário, o TensorFlow criou `GPU:0` e executou um `matmul` nela.

Em outras palavras: o estado da sessão gráfica/Metal e o domínio `launchd` no momento da inicialização do Python importam. Listar uma GPU não é prova suficiente; é preciso confirmar que uma operação real foi colocada em `GPU:0`.

## Evidências coletadas

| Verificação | Resultado | O que demonstra |
| --- | --- | --- |
| Hardware | Apple M4, 10 núcleos de GPU; `system_profiler` informa `Metal: Supported` | a GPU e o driver Metal estão presentes |
| Ambiente atual | CPython ARM64 3.12.3, TensorFlow 2.18.1, `tensorflow-metal` 1.2.0, Keras 3.15.1 | versões instaladas são coerentes |
| Ambiente da fase 1 | TensorFlow 2.18.1, `tensorflow-metal` 1.2.0, Keras 3.12.4 | o problema não é exclusivo do novo ambiente |
| Plug-in Metal | `libmetal_plugin.dylib` ARM64 presente nos dois ambientes; SHA-256 idêntico | não há cópia parcial ou binário de arquitetura errada |
| Dependências | `python -m pip check` sem requisitos quebrados | não há conflito declarado pelo instalador |
| Python no terminal integrado | `physical_gpus=[]`; `matmul` estrito aceita apenas CPU | o contexto que iniciou o run não tem Metal utilizável |
| PyTorch no mesmo contexto | MPS construído, mas indisponível | confirma que a limitação é da sessão/processo, não apenas de TensorFlow |
| Sessão gráfica | `launchctl gui/505` informa sessão `Aqua`; WindowServer em execução | há uma sessão gráfica de usuário ativa |
| Python reexecutado com `launchctl asuser 505` | `GPU:0` criada para Metal; `matmul` estrito retorna em `/device:GPU:0` | o mesmo ambiente funciona quando iniciado no domínio gráfico correto |

O log de sucesso do teste Aqua identificou explicitamente `Metal device set to: Apple M4` e a criação do dispositivo pluggable TensorFlow `GPU:0`.

## O que causou a diferença

O processo do treino atual (`PID 77085`) é filho de um `zsh` do terminal integrado do VS Code. O processo pai do VS Code foi observado em um caminho de App Translocation. Nesse contexto, a inicialização TensorFlow/Metal retornou `No supported GPU was found` e o processo continuou em CPU.

O mesmo interpretador, quando iniciado com `launchctl asuser 505`, usa o domínio da sessão Aqua ativa e passa no teste real de GPU. Isso comprova que o fator determinante é o contexto de execução/sessão, não o código da CNN ou as dependências do projeto.

Não foi possível atribuir, com os dados disponíveis, a origem interna exata da instabilidade — por exemplo, tela bloqueada anteriormente, sessão gráfica recém-restaurada ou estado herdado pelo terminal integrado. O App Translocation do VS Code é um fator de higiene a corrigir, mas não foi isolado como causa única.

## Por que o run atual continua em CPU

O run foi iniciado quando a GPU TensorFlow não era utilizável. A extração da CNN congelada aconteceu no início da execução; depois disso, a fase dominante é a QCNN.

A QCNN atual usa PennyLane `default.qubit`, `shots=None`, PyTorch em CPU e `float64`. O código rejeita tensores MPS para o circuito. Portanto:

- Metal só aceleraria a breve extração inicial das features pela CNN congelada;
- as épocas de aproximadamente 20 minutos da execução atual continuarão em CPU mesmo depois de restaurar Metal;
- não há ganho metodológico em reiniciar o run ativo somente para mover a QCNN a Metal, pois ela não possui esse caminho de execução.

## Correção implementada no repositório

### 1. Pré-verificação real de Metal

Foi adicionado [metal_preflight.py](src/quantum_models/metal_preflight.py). Ele falha se:

1. `tf.config.list_physical_devices("GPU")` estiver vazio; ou
2. um `matmul` pequeno com posicionamento estrito não retornar um tensor em `GPU:0`.

O segundo teste evita o falso positivo observado durante a investigação, em que uma enumeração transitória de GPU foi seguida por fallback para CPU.

### 2. Proteção no runner híbrido

[smoke.py](src/quantum_models/smoke.py) agora aceita `--require-tensorflow-gpu`. Com essa opção, a execução termina antes de carregar a CNN se o preflight falhar. O resultado final, quando houver sucesso, também registra a versão TensorFlow, GPU física e dispositivo do `matmul` de pré-verificação.

### 3. Launcher macOS/Aqua

Foi adicionado [run_hybrid_macos.sh](scripts/run_hybrid_macos.sh). Ele se reexecuta no domínio Aqua do usuário atual via `launchctl asuser`, roda o preflight e sempre inclui `--require-tensorflow-gpu`.

Validações realizadas após a alteração:

- `./scripts/run_hybrid_macos.sh --help` criou `GPU:0` Metal e passou no `matmul` estrito;
- `pytest -q`: **7 passed**;
- `python -m pip check`: sem requisitos quebrados;
- `compileall` de `src/` e `tests/`: concluído sem erros.

## Comando recomendado para a próxima rodada

Execute a partir da raiz do projeto. O launcher encerra imediatamente se Metal não estiver funcional:

```zsh
# Execute a partir da raiz do checkout.

run_stamp="$(date +%Y%m%d-%H%M%S)"
./scripts/run_hybrid_macos.sh \
  --dataset kmnist \
  --backend pennylane \
  --mode full \
  --head-epochs 50 \
  --head-batch-size 32 \
  --preferred-dtype float32 \
  --preferred-activation swish \
  --strict-model-profile \
  --seed 42 \
  --output-root "outputs/phase2-full-kmnist-50ep-${run_stamp}"
```

O primeiro bloco de saída precisa informar `Metal device set to: Apple M4` e `matmul_device` com `GPU:0`. Se não informar, não inicie a rodada.

## Recuperação quando o preflight falhar

1. Desbloqueie a sessão gráfica local e mantenha-a ativa.
2. Rode `./scripts/run_hybrid_macos.sh --help` novamente; ele é um teste sem treinamento.
3. Se ainda falhar, encerre totalmente o VS Code, mova-o de App Translocation para `/Applications` se necessário, abra-o de novo e repita o teste.
4. Se a GPU continuar indisponível após nova sessão, faça logout/login ou reinicie o Mac e repita o launcher antes de qualquer run longo.
5. Não reinstale TensorFlow ou `tensorflow-metal` como primeira medida: os dois ambientes e o plug-in binário já foram verificados e o mesmo conjunto funciona no domínio Aqua.

## Limites e próximos dados

O preflight foi validado, mas uma nova execução híbrida completa ainda não foi iniciada com a proteção. A próxima rodada deve preservar um novo `output-root` e guardar o `experiment-result.json` para documentar o dispositivo verificado junto com as métricas. A execução atual deve ser interpretada como ensaio CPU de integração; ela não mede desempenho Metal da CNN.

## Referência de compatibilidade

A Apple documenta que, para TensorFlow 2.13 ou posterior, a instalação esperada é `tensorflow` base com o plug-in `tensorflow-metal`; o Mac Apple Silicon com macOS 12+ e Python 3.9+ atende os requisitos publicados. [Apple — TensorFlow Metal](https://developer.apple.com/metal/tensorflow-plugin/)
