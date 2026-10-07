# Treinamentos encontrados nesta máquina e sincronização com os remotos

**Data da auditoria:** 30/09/2026 (BRT)<br>
**Cobertura:** os repositórios locais em `repos/teste-modelos-tcc` e seus subdiretórios Git, além de `Documents/Codex`. Foram encontrados registros de treinamento apenas nos três repositórios listados abaixo. O inventário usa arquivos de status, históricos, relatórios, manifests e o índice/histórico Git; diretórios descartados antes desta coleta não deixam evidência recuperável.

## Resumo

- Há **9 treinamentos Dense recentes**, um para cada dataset, todos concluídos em 100 épocas, seed 42. As métricas, curvas e telemetria compacta estão versionadas no remoto conforme a referência remota local. Os pesos finais e outros artefatos volumosos continuam somente no disco desta máquina.
- Há duas execuções híbridas QCNN/KMNIST: uma registrou **32 de 50 épocas** e outra não concluiu época alguma nos artefatos disponíveis. Ambas permanecem com status não terminal `running`; os diretórios estão locais e fora do Git.
- Nos experimentos históricos do repositório `teste-modelos-tcc`, encontrei campanhas de ativações, precisão/quantização, lotes e balanceamento. O inventário soma **142 arquivos de status** nessas campanhas: **141 `completed` e 1 `running` antigo**. Esse número inclui smoke runs e etapas de quantização e não equivale a 142 treinamentos longos independentes.
- As referências locais `origin/*` estão no mesmo commit que as branches de trabalho nos três repositórios (**0 commits à frente e 0 atrás**). Não consegui atualizar essas referências: `git fetch` falhou porque esta sessão não resolveu `github.com`. Portanto, “sincronizado” abaixo significa igual à referência `origin` já armazenada localmente, não uma confirmação ao vivo do estado atual do GitHub.

## Repositórios e sincronização Git

| Repositório | Branch | HEAD local = `origin` em cache | Estado observado |
|---|---|---|---|
| `projeto-tcc` | `codex/mac-imac-m4` | `91cf71b414ca6a6fd1be5287e9f4258d4014528f` | Branch sem commits pendentes; alterações locais de código e relatórios, além de resultados ignorados/não rastreados. |
| `teste-modelos-tcc/teste-modelos-tcc` | `codex/mac-training-split` | `f93c72270aba9ff1e85396e5949565f4027025e0` | Branch sem commits pendentes; artefatos rastreados estão no commit remoto em cache; alguns resultados ignorados são locais. |
| `teste-mac/tensorflow-metal-cuda-benchmark` | `results/mac` | `bc41d8c5c65a42c5feaa8af6f465b1136c5ba4ab` | Branch sem commits pendentes; há arquivos de smoke benchmark staged e um diretório de relatório não rastreado, ainda não publicados. Isso é benchmark, não treinamento de modelo. |

## Campanha recente: Dense sobre features congeladas

Executada em 29/09, com 9 modelos independentes, seed 42 e 100 épocas cada. O protocolo treinou uma cabeça `128 → 128 → 20 → classes` sobre vetores de 128 dimensões pré-computados; não fez fine-tuning da CNN. Todos os nove status indicam conclusão.

| Dataset | Acurácia de teste | Macro-F1 de teste |
|---|---:|---:|
| CIFAR-10 | 73,32% | 73,39% |
| CIFAR-100 coarse | 47,41% | 47,56% |
| EMNIST Balanced | 88,69% | 88,64% |
| Fashion-MNIST | 92,05% | 92,05% |
| FER2013 | 50,58% | 47,90% |
| GTSRB | 99,66% | 99,63% |
| KMNIST | 98,65% | 98,65% |
| MNIST | 99,23% | 99,23% |
| SVHN | 92,98% | 92,96% |

**Sincronização dos artefatos:** `reports/mac-training-data-2026-09-29-to-2026-09-30/` contém 65 arquivos rastreados no commit sincronizado em cache: históricos de 100 épocas, métricas de teste, relatórios por classe, tempos, estatísticas das features, resumos de telemetria e preflight. `outputs/classic-dense20/` contém manifests e vetores 128-D versionados via Git LFS. Já `outputs/classical-128-20-mac/` contém as execuções completas locais (incluindo pesos, previsões e outros arquivos), mas não tem arquivos rastreados no Git. Assim, **evidência resumida e features estão sincronizadas; checkpoints Dense e outputs integrais não estão**.

As mesmas nove pastas aparecem também sob `reports/mac-training-data...` e `outputs/classical-128-20-mac`; são cópias/visões da mesma campanha, não 18 treinamentos. O inventário não conta a extração de features como treinamento de classificador.

## QCNN híbrida: KMNIST

| Execução | Evidência disponível | Status/sincronização |
|---|---|---|
| `phase2-full-kmnist-50ep-20260928-174022` | `epoch-history.json` contém 32 épocas; a última amostra persistida é de 29/09 22:52 UTC. | `run-status.json` segue `running`; sem avaliação final de teste/status terminal. Somente local, sem arquivos rastreados. |
| `phase2-full-kmnist-50ep-20260929-214413` | `run-status.json` registra início em 29/09 21:44 BRT (30/09 00:44 UTC); há telemetria inicial, mas nenhuma época concluída nem resultado final nos artefatos inspecionados. | `running` no último status; atividade atual do processo não pôde ser confirmada nesta sessão. Somente local. |

O registro anterior da primeira execução, em 28/09, mostrava cinco épocas; o histórico atual avançou até 32. A descrição detalhada do protocolo e as limitações de comparabilidade estão em `RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md`. Os modelos clássicos FP32/FP16 que alimentam a fase híbrida foram importados do conjunto de quantização histórico; são os mesmos 18 modelos de origem, não uma campanha adicional nesta contagem.

## Campanhas históricas no repositório `teste-modelos-tcc`

Os caminhos abaixo estão dentro de `teste-modelos-tcc/teste-modelos-tcc/outputs/`. A contagem foi feita a partir de `status.json`; o estado “remoto” descreve presença de arquivos de evidência rastreados, não necessariamente os checkpoints completos.

| Campanha | Execuções/status encontrados | Sincronização com `origin` em cache |
|---|---|---|
| Ativações com augmentation 0,5 (`controlled-augmentation05-activations-mac2`) | 27/27 concluídas, 3 ativações × 9 datasets. | Resumo consolidado de métricas está versionado; 18 dos 27 status por run estão rastreados, os outros 9 e parte dos outputs são locais. |
| Treinamentos da campanha de quantização (`controlled-quantization-fast-mac-m4`) | 18 treinamentos FP32/FP16 concluídos (9 datasets × 2 precisões); mais 9 etapas de quantização associadas aparecem como concluídas. | Relatórios consolidados estão versionados. Os 27 diretórios de status/resultados dessa pasta estão locais e não rastreados. |
| Smoke da campanha de quantização | 9/9 concluídos. | Somente local/não rastreado. |
| Ativações com augmentation 2,0 (`controlled-augmentation2-activations-mac2`) | 5 concluídas; 1 status antigo ainda `running`; 9 smoke runs concluídos. | Nenhum status dessa família está rastreado; outputs locais. O status `running` é de 28/08 e não prova processo ainda ativo. |
| Smoke e varredura de batch com augmentation 0,5 (`controlled-augmentation2-mac-m4-aug05`) | 9 smoke runs e 6 runs de batch concluídos. | Os 15 status estão rastreados. |
| Varredura KMNIST sem augmentation, batch 16–512 (`kmnist-alldata-noaug-batch-sweep-2026-08-06`) | 32 execuções, todas concluídas, incrementos de batch de 16. | Status e evidência rastreados; sincronizados com a referência local de `origin`. |
| Experimentos de balanceamento (`remaining-ram-capped`) | 12 execuções concluídas: FER2013, SVHN e GTSRB, com quatro estratégias por dataset. | Status rastreados e sincronizados com a referência local. |
| Variações de precisão/quantização de KMNIST (`kmnist-quantization-2026-08-09`) | 5 registros concluídos: FP32, FP16, INT8 PTQ, INT8 QAT e INT4 QAT. | Status rastreados e sincronizados com a referência local. |

O relatório consolidado versionado `analysis_reports/treinamento_consolidado_2026-09-14/README.md` registra a campanha 0,5 com 27/27 resultados e a etapa FP32/FP16 com 18/18; também resume as conversões/avaliações de quantização. Esse relatório é a fonte resumida no remoto para execuções cujos diretórios brutos foram mantidos localmente.

## O que está e o que não está no remoto

- **Sincronizados segundo o `origin` em cache:** os três commits atuais; os relatórios consolidados da campanha de ativações e quantização; os registros completos da varredura KMNIST, das 12 execuções de balanceamento e das cinco variações KMNIST; os 15 status da campanha de batch 0,5; os nove relatórios compactos Dense recentes e os vetores/manifests de features via Git LFS.
- **Parcialmente sincronizados:** resultados detalhados da campanha 0,5 de ativações (18/27 status rastreados, com o restante local); modelos FP32/FP16 e 9 etapas de quantização (resumo remoto, diretórios de execução não rastreados); Dense recente (métricas e históricos remotos, checkpoints e outputs integrais somente locais).
- **Somente locais:** as duas execuções QCNN incompletas; outputs da família augmentation 2,0; smokes da família de quantização; outputs completos da campanha Dense recente.
- **Ainda não publicados:** alterações staged/untracked no benchmark `tensorflow-metal-cuda-benchmark`; são resultados de benchmark de hardware, não treinamentos de modelo.

## Limitações da auditoria

1. O remoto foi comparado às referências `origin/*` presentes no clone. A atualização via rede falhou com `Could not resolve host: github.com`; o estado do GitHub pode ter mudado desde o último fetch local.
2. O comando de inspeção de processos do sistema foi bloqueado por permissão (`ps: operation not permitted`). Para as duas QCNN, o relatório registra o último estado persistido; não afirma que ainda havia processo ativo.
3. Resultados apagados, armazenados fora dos diretórios de trabalho examinados, ou em volumes externos não são detectáveis por este inventário. A busca cobriu `~/repos` e `~/Documents/Codex`; encontrou os arquivos de status somente na família de repositórios acima.
4. Smoke tests e etapas de conversão/quantização aparecem separadas de treinamentos completos para evitar contar cada registro técnico como uma rodada longa independente.

## Fontes locais

- `RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md` — relatório completo da campanha Dense recente e das duas tentativas QCNN.
- `reports/mac-training-data-2026-09-29-to-2026-09-30/README.md` — inventário dos 65 arquivos de evidência compacta que acompanham os resultados recentes.
- `../../teste-modelos-tcc/analysis_reports/treinamento_consolidado_2026-09-14/README.md` — resumo histórico das ativações e quantização.
- Status por execução em `../../teste-modelos-tcc/outputs/` e `outputs/`.
