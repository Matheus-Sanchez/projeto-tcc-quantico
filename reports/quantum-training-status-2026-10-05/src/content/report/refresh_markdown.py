from pathlib import Path
import json
from datetime import datetime
from zoneinfo import ZoneInfo
p=Path('src/data.json');d=json.loads(p.read_text());q={k:v['rows'] for k,v in d['queries'].items()};o=q['overview'][0];runs=q['runs'];done=[r for r in runs if r['status']=='completed'];started=[r for r in runs if r['confirmedEpochs']];active=next((r for r in runs if r['status']=='running'),None);em=next(r for r in runs if r['dataset']=='emnist_balanced');cifars=[r for r in done if r['dataset'] in ['cifar10','cifar100_coarse']]
def num(v):return f'{v:,}'.replace(',','.')
def pct(v):return '—' if v is None else f'{100*v:.2f}%'.replace('.',',')
def dec(v,n=4):return f'{v:.{n}f}'.replace('.',',')
def moment(v):return datetime.fromisoformat(v).astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y às %Hh%M')
def hours(v):
 m=round(v*60);return f'{m//60}h{m%60:02}min'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
all_complete = len(done)==len(runs)
latest_completion = max((r['completedAt'] for r in done),default=None)
process_sentence = 'O processo estava ativo durante a coleta.' if o['activeProcess'] else 'Nenhum processo de treinamento estava ativo na leitura.'
status_sentence = f"A execução terminou em {moment(latest_completion)}, com treino, teste e estudo de ruído concluídos nos nove datasets." if all_complete else (f"O {active['label']} segue em execução." if active else 'Não há treinamento ativo na leitura.')
pending = [r['label'] for r in runs if r['status']=='pending']
queue_sentence = 'Na fila: '+', '.join(pending)+'.' if pending else 'Não há datasets aguardando na fila.'
batch_sentence = f"Inclui lotes parciais do {active['label']}." if active else 'Todos os lotes pertencem a épocas completas.'
progress_sentence = f"Todas as {o['plannedEpochs']} épocas planejadas foram concluídas (100%). Não há treino, teste ou estudo de ruído pendente." if all_complete else f"{o['confirmedEpochs']}/{o['plannedEpochs']} épocas confirmadas representam {pct(o['confirmedEpochs']/o['plannedEpochs'])} da contagem planejada, sem ponderar tempo ou custo por dataset. Não é uma previsão de duração restante."
content=f'''# Relatório do treinamento quântico com Qiskit

Leitura: {moment(o['observedAt'])}, America/Sao_Paulo. {process_sentence} Este relatório registra uma leitura pontual.

**{o['completedRuns']} concluídos, {o['activeRuns']} em execução e {o['pendingRuns']} aguardando.** {status_sentence} {queue_sentence} Os nove datasets tiveram suas features validadas antes do primeiro treinamento.

{table(['Dataset','Etapa','Épocas confirmadas','Melhor época','Acurácia de teste','Macro-F1 de teste'],[[r['label'],r['stage'],str(r['confirmedEpochs'])+'/100',r['bestEpoch'] or '—',pct(r['testAccuracy']),pct(r['testMacroF1'])] for r in runs])}

## CIFAR: novos resultados de teste

'''
for r in cifars:
 content+=f"**{r['label']}** terminou em {moment(r['completedAt'])}, após {hours(r['durationHours'])}. O checkpoint da época {r['bestEpoch']} foi selecionado pelo macro-F1 de validação de {pct(r['bestValMacroF1'])}. Nos {num(r['test'])} exemplos de teste, a acurácia foi **{pct(r['testAccuracy'])}** e o macro-F1 foi **{pct(r['testMacroF1'])}**. Foram concluídas 100 épocas e 64 condições de ruído. Na última época, a diferença entre acurácia de treino e validação foi de {dec((r['lastTrainAccuracy']-r['lastValAccuracy'])*100,2)} pontos percentuais.\n\n"
content+=f'''## EMNIST Balanced concluído

A execução terminou em {moment(em['completedAt'])}, após {hours(em['durationHours'])}. O checkpoint da época {em['bestEpoch']} foi selecionado pelo macro-F1 de validação de {pct(em['bestValMacroF1'])}. No conjunto completo de {num(em['test'])} exemplos de teste, a acurácia foi **{pct(em['testAccuracy'])}** e o macro-F1 foi **{pct(em['testMacroF1'])}**. Foram salvas predições, métricas por classe, matriz de confusão e 64 condições de ruído.

'''
if active:
 content+=f'''## {active['label']} em execução

**{active['confirmedEpochs']} épocas confirmadas; época {active['currentEpoch']} em andamento**, com {active['currentSavedBatches']} de {active['batchesPerEpoch']} lotes de treino salvos no instante da leitura. A retomada usa o checkpoint da última época completa.

Na época {active['confirmedEpochs']}: acurácia de treino {pct(active['lastTrainAccuracy'])}, loss {dec(active['lastTrainLoss'])}; acurácia de validação {pct(active['lastValAccuracy'])}, macro-F1 {pct(active['lastValMacroF1'])}, loss {dec(active['lastValLoss'])}. O melhor macro-F1 de validação é **{pct(active['bestValMacroF1'])}, na época {active['bestEpoch']}**. Ainda não foi coletado teste final.

A diferença entre acurácia de treino e validação na última época é de {dec((active['lastTrainAccuracy']-active['lastValAccuracy'])*100,2)} pontos percentuais. O aumento da acurácia de treino não indica melhora equivalente na validação; o melhor checkpoint permanece anterior à última época.

A sessão tinha {hours(active['durationHours'])} de duração. A mediana das últimas cinco épocas é de {dec(active['recentEpochSeconds']/60,1)} minutos para treino e validação; a consolidação de diagnósticos e checkpoint ocorre depois. Faltam {100-active['confirmedEpochs']} épocas confirmadas, além do teste e estudo de ruído.

'''
else:
 content+='## SVHN, GTSRB e FER2013: resultados finais\n\n'
 for r in done:
  if r['dataset'] in ['svhn','gtsrb','fer2013']:
   content+=f"**{r['label']}** terminou em {moment(r['completedAt'])}, após {hours(r['durationHours'])}. O checkpoint da época {r['bestEpoch']} foi escolhido pelo macro-F1 de validação de {pct(r['bestValMacroF1'])}. Nos {num(r['test'])} exemplos de teste, a acurácia foi **{pct(r['testAccuracy'])}** e o macro-F1 foi **{pct(r['testMacroF1'])}**. Foram concluídas 100 épocas e 64 condições de ruído.\n\n"
 fer=next(r for r in runs if r['dataset']=='fer2013')
 content+=f"No FER2013, a última época registrou treino {pct(fer['lastTrainAccuracy'])} e validação {pct(fer['lastValAccuracy'])}, diferença de {dec((fer['lastTrainAccuracy']-fer['lastValAccuracy'])*100,2)} pontos percentuais. A acurácia mais alta do treino não se mantém na validação. O teste usa o checkpoint da época {fer['bestEpoch']}, selecionado pela validação.\n\n"
content+=f'''## Dados coletados

- {num(o['confirmedEpochs'])} registros de épocas completas, com loss, acurácia, macro-F1, métricas por classe e duração.
- {num(o['trainBatchJson'])} JSONs de lotes de treino e {num(o['validationBatchJson'])} de validação; quatro registros de circuito por arquivo, totalizando {num(o['batchCircuitRecords'])} registros. {batch_sentence}
- {num(o['telemetrySamples'])} amostras de telemetria (CPU, RAM e disco).
- Checkpoints, sondas de estados, gradientes, alterações de pesos, tempos por circuito e contagem de avaliações lógicas nos {len(started)} treinamentos iniciados.
- Predições de teste, métricas por classe e matrizes de confusão nos {len(done)} concluídos, cobrindo {num(o['testExamplesCompleted'])} exemplos de teste.
- {num(o['noiseConditions'])} condições do estudo de ruído/shots: 64 por dataset concluído.

{table(['Dataset','JSONs de treino','JSONs de validação','Sondas','Condições de ruído','Predições de teste'],[[r['label'],num(r['trainBatchJson']),num(r['validationBatchJson']),r['probeJson'],r['noiseConditions'],'Salvas' if r['testPredictions'] else 'Ainda não'] for r in q['inventory']])}

## Tempo e telemetria

{table(['Dataset','Tempo da sessão','Amostras','Pico RAM da árvore','Última amostra'],[[r['label'],hours(r['elapsedHours']),num(r['samples']),dec(r['peakMemoryMbTree']/1024,2)+' GiB',moment(r['lastAt'])] for r in q['telemetry']])}

## Estudo de ruído

Escalas 0, 0,5, 1 e 2; leitura exata e 256/1.024/4.096 shots; cinco seeds (42 a 46) por condição finita. O estudo reutiliza o melhor checkpoint e simula todo o teste localmente.

Na escala 2 com 256 shots (queda negativa indica aumento da acurácia):

{table(['Dataset','Acurácia média','Mínimo','Máximo','Queda média (p.p.)'],[[r['label'],pct(r['accuracy']),pct(r['minAccuracy']),pct(r['maxAccuracy']),dec(r['degradationPp'],3)] for r in q['noise_aggregate'] if r['noiseScale']==2 and r['shots']==256])}

Mínimo e máximo descrevem as cinco seeds de amostragem, sem representar um intervalo de confiança. Essas seeds não são repetições independentes do treinamento.

## Features disponíveis

{table(['Dataset','Classes','Treino','Validação','Teste'],[[r['label'],r['classes'],num(r['train']),num(r['val']),num(r['test'])] for r in q['features']])}

São {num(o['totalInputExamples'])} exemplos entre todos os splits, cada um com 128 valores float32. Nos {len(started)} datasets iniciados, as estatísticas salvas registram zero NaN e zero infinito.

## Protocolo e interpretação

Cabeça: features128 → Dense(128, ReLU) → Dense(20, linear) → quatro circuitos de cinco qubits → logits. São 180 parâmetros quânticos e 19.482 parâmetros treináveis para dez classes. Qiskit Aer local em CPU, quatro workers persistentes, Adam, learning rate 0,0003, batch 128, seed 42 e 100 épocas. Sem parada antecipada ou redução automática da taxa de aprendizado.

Os checkpoints usados no teste foram selecionados pelo maior macro-F1 de validação: {', '.join('época '+str(r['bestEpoch'])+' ('+r['label']+')' for r in done)}. No Fashion-MNIST, a última época tem treino 99,95% e validação 91,97%; a avaliação final utiliza a época 5.

{progress_sentence}

A referência clássica salva nos {len(started)} runs é inelegível: faltam status concluído e métricas de teste naquela referência. Este relatório não estabelece ganho quântico sobre o clássico. Há um único seed de treinamento por dataset.

## Fontes e verificações

Arquivos locais em outputs/quantum-parallel-128-20/runs: matrix.json; dataset/qiskit/seed-42/{{status.json, checkpoints/state.json, history.csv, test_metrics.json, result.json, config.json, feature_statistics.json}}; diagnostics/{{batches, probes, noise-study}}; telemetry/*/samples.jsonl. Features em outputs/classic-dense20/*/runs/*/artifacts/features128.

O coletor está em collect_evidence.py, neste diretório. Os 54 cabeçalhos NPY foram conferidos. Históricos e melhor macro-F1 foram reconciliados com os checkpoints; métricas de teste foram reconciliadas com os {len(done)} resultados. Configurações e estatísticas de features foram conferidas nos {len(started)} runs iniciados. {batch_sentence} O relatório interativo mantém as fontes e definições nos menus de cada componente.
'''
Path('src/content/report/relatorio.md').write_text(content)
print('Relatório em texto atualizado:',len(content),'caracteres')
