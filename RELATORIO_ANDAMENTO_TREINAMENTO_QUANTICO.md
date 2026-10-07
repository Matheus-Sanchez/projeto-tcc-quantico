# Andamento do treinamento híbrido QCNN — KMNIST

**Corte dos dados:** 2026-09-28 19:38:41 BRT (22:38:41 UTC)<br>
**Execução:** `outputs/phase2-full-kmnist-50ep-20260928-174022/kmnist/full__pennylane__float32_swish_all_raw__seed-42/`<br>
**Status no corte:** `running`; cinco de 50 épocas concluídas; a sexta estava em execução.
**Escopo deste documento:** evidência já gravada no disco até o corte. Métricas de teste e conclusão do experimento ainda não existem.

## Resumo executivo

- O treinamento está **ativo e saudável no sentido operacional**: o processo `77085` estava em estado `R+`, consumindo cerca de um núcleo de CPU, e a telemetria continuava sendo atualizada a cada cinco segundos. Não há sinal de interrupção, falta de espaço ou crescimento anômalo de memória.
- Nas cinco épocas já concluídas, o melhor checkpoint é o da época 5. O macro-F1 de validação aumentou de **0,9880** para **0,9906** e a perda de validação caiu de **0,1092** para **0,0369**. É um sinal inicial consistente de aprendizado, mas ainda é cedo para afirmar desempenho final ou superioridade quântica.
- A execução é de fato híbrida: usa a CNN clássica FP32/SwiSH importada e congelada para gerar 256 características, depois treina uma QCNN PennyLane de 8 qubits e o classificador denso final. A CNN não recebe gradientes.
- A execução **não está usando TensorFlow Metal**: a pré-verificação no mesmo ambiente retornou `physical_gpus=[]`. Portanto, ela comprova a viabilidade funcional em CPU, mas não valida desempenho da fase TensorFlow no GPU Apple M4.
- Há uma limitação metodológica relevante: a fase 2 divide apenas os 60.000 exemplos oficiais de treino, enquanto a CNN da fase 1 foi treinada em uma divisão de 70.000 exemplos combinados. A reconstrução determinística mostrou que 58,7% da validação e 63,3% do teste da fase 2 já haviam sido vistos pela CNN congelada no treino da fase 1. Este run é adequado como teste end-to-end; não é ainda uma comparação independente entre CNN clássica e cabeça quântica.

## Configuração realmente em execução

| Item | Valor observado |
| --- | --- |
| Dataset | KMNIST, 10 classes (`0`–`9`) |
| Backend quântico | PennyLane `default.qubit`, simulador exato (`shots=None`) |
| Modo | `full` |
| Seed | 42 |
| Épocas solicitadas | 50 |
| Batch da cabeça | 32 |
| Otimizador / taxa | Adam / `0,01` |
| Avaliação de validação | a cada época |
| Perfil de CNN exigido | `float32`, ativação `swish`, `all_raw`, `unit_interval`, seleção estrita |
| CNN de origem | fase 1 concluída, `kmnist/fp32`, checkpoint `best.keras` |
| Camada congelada | `global_pool_concat`, vetor de 256 dimensões, zero variáveis treináveis |
| Parte treinável | 51 parâmetros quânticos e `Linear(256, 10)`; 2.621 parâmetros no total |
| Codificação / medição | amplitude encoding L2 normalizado em 8 qubits; probabilidades dos 256 estados |
| Precisão da QCNN | CPU `float64` |

A escolha do artefato clássico foi validada pelo manifesto da fase 1: `status: completed`, `all_raw`, `unit_interval`, 10 classes, FP32 e SwiSH. A CNN clássica correspondente foi treinada por 100 épocas, com `extra_fraction: 0,5` na fase 1.

## Dados e partições usados nesta execução

O audit local do KMNIST leu **70.000** exemplos: 60.000 no split oficial de treino e 10.000 no split oficial de teste. Encontrou 10 classes balanceadas, imagens `28x28x1` em `uint8`, sem rótulos/imagens inválidos, arquivos ausentes ou avisos.

O modo `full` da fase 2 usa somente o split oficial `train` e reaplica uma divisão estratificada 70/15/15 com seed 42:

| Partição da fase 2 | Exemplos | Por classe | Uso já concluído |
| --- | ---: | ---: | --- |
| Treino QCNN | 42.000 | 4.200 | cinco passagens completas |
| Validação | 9.000 | 900 | medida a cada época |
| Teste final | 9.000 | 900 | ainda não avaliado |
| Split oficial `test` do KMNIST | 10.000 | 1.000 | não utilizado por este run |

Foram executadas 1.313 atualizações do otimizador por época. Assim, até a época 5 foram processadas 210.000 apresentações de treino e 45.000 de validação, sem contar as avaliações internas da diferenciação automática.

### Qualidade e procedência já coletadas

- Audit: `outputs/phase2-input-audit/kmnist/audit/audit.json` — saudável, zero erros e zero avisos.
- A verificação de formas, canais e rótulos cobriu os 70.000 exemplos.
- A opção de hash de pixels estava desativada no audit; portanto, a ausência de duplicatas por conteúdo **não foi testada** neste corte. Isso não é um erro do dataset, apenas uma verificação ainda ausente.
- O checkpoint e seu `manifest.json` foram copiados da fase 1 com verificação SHA-256 antes desta execução.

## Série de métricas já coletada

Todas as métricas abaixo vêm de `epoch-history.json`; não há interpolação.

| Época | Loss treino | Acc. treino | Macro-F1 treino | Loss validação | Acc. validação | Macro-F1 validação | Tempo | Vazão |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,5039 | 0,9800 | 0,9801 | 0,1092 | 0,9880 | 0,9880 | 20m56s | 33,43 ex/s |
| 2 | 0,0683 | 0,9930 | 0,9930 | 0,0566 | 0,9897 | 0,9897 | 20m37s | 33,95 ex/s |
| 3 | 0,0419 | 0,9933 | 0,9933 | 0,0431 | 0,9901 | 0,9901 | 20m30s | 34,15 ex/s |
| 4 | 0,0332 | 0,9935 | 0,9935 | 0,0378 | 0,9900 | 0,9900 | 20m33s | 34,06 ex/s |
| 5 | 0,0292 | 0,9937 | 0,9937 | 0,0369 | 0,9906 | **0,9906** | 20m34s | 34,04 ex/s |

Do primeiro ao quinto ponto de medição:

- a perda de treino caiu 94,2% (0,5039 → 0,0292);
- a perda de validação caiu 66,2% (0,1092 → 0,0369);
- a acurácia de validação aumentou 0,26 ponto percentual (98,80% → 99,06%);
- o macro-F1 de validação aumentou 0,25 ponto percentual (0,9880 → 0,9906);
- a diferença de acurácia entre treino e validação na época 5 é 0,32 ponto percentual. Não há, neste corte, evidência forte de sobreajuste, mas cinco épocas não permitem concluir estabilidade até a época 50.

### Melhor checkpoint disponível

`qcnn-head-best.pt` e `qcnn-head-last.pt` apontam para a época 5. O critério de seleção é: maior macro-F1 de validação e, em caso de empate, menor perda de validação.

| Métrica de validação do checkpoint da época 5 | Valor |
| --- | ---: |
| Loss de entropia cruzada | 0,036899 |
| Acurácia / acurácia balanceada | 0,990556 / 0,990556 |
| Macro-precision | 0,990587 |
| Macro-recall | 0,990556 |
| Macro-F1 | 0,990556 |

Na validação da época 5, o menor F1 por classe foi da classe `1` (**0,9827**) e o maior foi da classe `9` (**0,9967**).

| Classe | F1 de validação — época 5 |
| ---: | ---: |
| 0 | 0,9899 |
| 1 | 0,9827 |
| 2 | 0,9900 |
| 3 | 0,9961 |
| 4 | 0,9906 |
| 5 | 0,9939 |
| 6 | 0,9862 |
| 7 | 0,9928 |
| 8 | 0,9867 |
| 9 | 0,9967 |

### Sinais do circuito e dos gradientes

Na época 5, as 9.000 medições de validação tiveram soma média de probabilidades igual a 1,0000000000000004, e o maior erro de normalização foi `3,9968e-15`. Isso é consistente com uma distribuição de probabilidades numericamente válida.

| Indicador quântico — época 5 | Valor |
| --- | ---: |
| Estados de base efetivos, média | 72,45 de 256 |
| Entropia média | 4,2773 nats |
| Maior probabilidade média | 0,0810 |
| Menor probabilidade observada | `2,34e-12` |
| Norma L2 média do gradiente de `theta` | 0,1045 |
| Maior norma L2 do gradiente de `theta` | 0,6428 |
| Norma L2 da atualização de `theta` | 0,3445 |

Os gradientes são finitos e não nulos. Não há evidência de barren plateau nas cinco épocas observadas; também não há evidência suficiente para afirmar que ele não ocorrerá mais adiante.

## Recursos e previsão operacional

| Indicador | Evidência no corte |
| --- | --- |
| Processo | PID 77085, estado `R+` e 98,5% de CPU na inspeção direta |
| Telemetria mais recente | processo em 99,8% de CPU, 34 threads, RSS 1,03 GiB |
| Fim de época | RSS entre 1,36 e 1,37 GiB; memória do sistema entre 70,4% e 71,2% |
| CPU do sistema | 13,2% a 22,7% em uma máquina de 10 núcleos lógicos; a simulação é predominantemente serial |
| Espaço livre | cerca de 226,9 GiB; sem risco de disco no horizonte desta execução |
| GPU do TensorFlow | `[]` no preflight atual; `No supported GPU was found` |

O tempo por época ficou entre 20m30s e 20m56s, com média de 20m38s. Mantido esse ritmo, as 50 épocas mais a avaliação final devem encerrar aproximadamente entre **10:50 e 11:10 BRT de 2026-09-29**. É uma projeção por média de cinco épocas, não uma garantia.

O simulador PennyLane é propositalmente executado em CPU e `float64`; por isso a ausência de Metal não impede a QCNN. Entretanto, a extração inicial de características pela CNN também ocorreu com TensorFlow sem GPU. O run valida que a cadeia funciona, mas não mede o desempenho pretendido de uma CNN em Metal.

## Comparabilidade com a fase 1

O melhor modelo clássico importado para o KMNIST reporta, na fase 1, macro-F1 de teste **0,9819** em 10.500 exemplos. Esse número **não deve ser comparado diretamente** ao macro-F1 de validação de 0,9906 da fase 2, pelos motivos abaixo:

1. A fase 1 dividiu os 70.000 exemplos combinados em 49.000/10.500/10.500; a fase 2 dividiu somente os 60.000 de treino oficial em 42.000/9.000/9.000.
2. A fase 1 aplicou augmentation com `extra_fraction: 0,5`; a fase 2 chama o pipeline com `training=False` e `extra_fraction: 0,0`. Não há augmentation durante a extração de features desta execução.
3. O código de divisão é determinístico e a reconstrução reproduziu o fingerprint da fase 1 (`161c6421…`). Nessa reconstrução, a CNN congelada havia visto no treino da fase 1:

| Partição da fase 2 | Exemplos vistos antes pela CNN da fase 1 | Fração |
| --- | ---: | ---: |
| Treino | 31.023 de 42.000 | 73,9% |
| Validação | 5.284 de 9.000 | 58,7% |
| Teste | 5.700 de 9.000 | 63,3% |

Isso é aceitável para um ensaio de integração que testa a cabeça sobre features de uma CNN já treinada. Para alegar desempenho preditivo independente ou comparar arquiteturas, a avaliação precisa usar amostras que não tenham participado do treino da CNN congelada, preferencialmente preservando um único fingerprint de split para todos os braços.

## Artefatos presentes e artefatos ainda esperados

| Artefato | Estado |
| --- | --- |
| `run-status.json` | presente, `running` |
| `epoch-history.json` | presente, cinco registros completos |
| `qcnn-head-best.pt` | presente, checkpoint da época 5 |
| `qcnn-head-last.pt` | presente, checkpoint da época 5 |
| `telemetry/hardware.json` e `telemetry/samples.jsonl` | presentes; amostras periódicas ativas |
| `experiment-result.json` | ausente — só é criado no encerramento bem-sucedido |
| `validation-report/` | ausente — só é produzido no encerramento |
| `test-report/` | ausente — avaliação de teste ainda não ocorreu |
| `telemetry/summary.json` | ausente — criado quando a telemetria é encerrada |

## Dados que ainda faltam coletar

### Necessários para concluir este run

1. As 45 épocas restantes, incluindo a evolução após a aparente estabilização inicial.
2. A escolha final do melhor checkpoint, que pode mudar após a época 5.
3. Métricas de teste em 9.000 exemplos, matriz de confusão, relatórios por classe e `experiment-result.json`.
4. Contagens finais de avaliações de circuito, batches e parâmetros persistidas no resultado final.
5. Sumário de telemetria final e o status terminal `passed` ou `failed`.

### Necessários para uma conclusão científica comparativa

1. Uma nova execução híbrida com split independente da CNN congelada, ou uma CNN treinada somente no treino desse mesmo split. O teste oficial KMNIST de 10.000 exemplos é uma opção simples de holdout, desde que a CNN também não o tenha usado.
2. Uma execução fase 2 que aplique a política de augmentation `0,5` de modo explicitamente controlado, em um novo diretório de saída. O run atual não a aplica.
3. Uma cabeça clássica densa treinada sobre as **mesmas** features, split, seed, batch, otimizador e épocas. Ela é o baseline necessário para isolar o efeito da QCNN.
4. Repetições com sementes adicionais e intervalo de confiança/variabilidade entre execuções; há apenas uma seed até agora.
5. A contraparte Qiskit nas mesmas condições, se o objetivo incluir comparação entre backends quânticos.
6. Um preflight bem-sucedido de TensorFlow Metal e uma medição separada de desempenho da extração CNN em GPU. O run atual não fornece essa evidência.
7. Um audit complementar com hash de pixels e verificação de imagens, se for necessário documentar integralmente a qualidade dos dados no TCC.

## Conclusão do corte

O ensaio está progredindo normalmente e já demonstrou aprendizado mensurável da cabeça híbrida. A melhor validação atual é 99,06% de acurácia e macro-F1, com gradientes quânticos estáveis. Ainda não existe métrica de teste nem base metodológica para afirmar que a QCNN supera a CNN clássica: faltam o fim do run, uma avaliação de holdout independente e um baseline denso pareado.

## Evidências primárias

- `outputs/phase2-full-kmnist-50ep-20260928-174022/kmnist/full__pennylane__float32_swish_all_raw__seed-42/run-status.json`
- `outputs/phase2-full-kmnist-50ep-20260928-174022/kmnist/full__pennylane__float32_swish_all_raw__seed-42/epoch-history.json`
- `outputs/phase2-full-kmnist-50ep-20260928-174022/kmnist/full__pennylane__float32_swish_all_raw__seed-42/telemetry/samples.jsonl`
- `outputs/phase2-input-audit/kmnist/audit/audit.json`
- `models/controlled-quantization-fast-mac-m4/quantization/kmnist/fp32/kmnist/runs/kmnist__unit_interval__all_raw__seed-42/manifest.json`
- `src/quantum_models/smoke.py` e `src/quantum_models/qcnn.py`
