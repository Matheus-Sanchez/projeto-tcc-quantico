"""Read-only snapshot of the local Qiskit training matrix for a report."""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT = Path('/Users/matheus-duda/Desktop/repos/projeto-do-tcc/projeto-tcc-quantico')
sys.path.insert(0, str(PROJECT / 'src'))
from data_prep.feature_vectors import discover_features128
from data_prep.registry import DATASET_ORDER
from utils.experiment_config import DEFAULT_FEATURES_ROOT, NUM_CLASSES

TZ = ZoneInfo('America/Sao_Paulo')
ROOT = PROJECT / 'outputs/quantum-parallel-128-20/runs'
LABELS = dict(zip(DATASET_ORDER, ['MNIST','Fashion-MNIST','KMNIST','EMNIST Balanced','CIFAR-10','CIFAR-100 Coarse','SVHN','GTSRB','FER2013']))

def read(p, default=None):
    return json.loads(p.read_text()) if p.is_file() else ({} if default is None else default)

def relative(p):
    return str(p.relative_to(PROJECT))

def local(iso):
    return datetime.fromisoformat(iso).astimezone(TZ).isoformat() if iso else None

def source(label, files, definitions, caveats):
    return {'label': label, 'provider': 'Arquivos locais de treinamento', 'files': list(dict.fromkeys(files)),
            'queriedAt': observed, 'timezone': 'America/Sao_Paulo',
            'filters': ['Qiskit; seed 42; modo full; matriz sequencial de nove datasets'],
            'caveats': caveats, 'metricDefinitions': definitions,
            'evidenceFlow': [{'title': 'Leitura local', 'detail': 'JSON, CSV e cabeçalhos NPY lidos diretamente, sem alterar o treinamento.'},
                             {'title': 'Progresso confirmado', 'detail': 'checkpoints/state.json governa épocas confirmadas; arquivos por lote identificam a época parcial. Valores ausentes são null.'}]}

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
observed = datetime.now(timezone.utc).isoformat()
matrix = read(ROOT/'matrix.json')
runs, features, history, inventory, noise, telemetry, checks = [], [], [], [], [], [], []
run_files, history_files, feature_files, noise_files, inventory_files, telemetry_files = [], [], [], [], [], []
processes = subprocess.check_output(['ps','-axo','pid,ppid,etime,time,%cpu,%mem,state,command'], text=True)
active = [p.strip() for p in processes.splitlines() if '-m quantum_models.parallel_dense20 --all --backend qiskit --resume' in p and p.split()[-1] == '--resume' and 'python' in p]

for ds in DATASET_ORDER:
    d = ROOT/ds/'qiskit/seed-42'
    j = next(j for j in matrix['jobs'] if j['dataset'] == ds and j['backend'] == 'qiskit')
    config = j['configuration']
    folder = discover_features128(ds, DEFAULT_FEATURES_ROOT)
    stat = read(d/'feature_statistics.json').get('statistics', {})
    f = {'dataset': ds, 'label': LABELS[ds], 'classes': NUM_CLASSES[ds], 'width': 128, 'dtype': 'float32',
         'source': relative(folder), 'validatedByMatrix': True, 'statisticsRecorded': bool(stat)}
    for split in ('train','val','test'):
        xp, yp = folder/f'{split}_features.npy', folder/f'{split}_labels.npy'
        x, y = np.load(xp, mmap_mode='r', allow_pickle=False), np.load(yp, mmap_mode='r', allow_pickle=False)
        assert x.shape[1:] == (128,) and x.dtype == np.float32 and len(x) == len(y)
        f[split] = len(x)
        feature_files.extend([relative(xp), relative(yp)])
        if stat:
            assert stat[split]['examples'] == len(x) and stat[split]['nan_count'] == stat[split]['inf_count'] == 0
    f['total'] = f['train'] + f['val'] + f['test']
    features.append(f)
    status = read(d/'status.json'); state = read(d/'checkpoints/state.json'); result = read(d/'result.json')
    epochs = int(state.get('epoch', 0)); planned = config['effective_epochs']
    rows = list(csv.DictReader((d/'history.csv').open())) if (d/'history.csv').exists() else []
    # Only committed full epochs belong to the authoritative history.
    committed = [r for r in rows if int(r['epoch']) <= epochs]
    hist = []
    for h in committed:
        hrow = {'dataset': ds, 'label': LABELS[ds]}
        for k in ('epoch','loss','accuracy','val_loss','val_accuracy','val_macro_f1','epoch_seconds','train_examples','train_examples_per_second','learning_rate'):
            hrow[k] = float(h[k]) if h.get(k) else None
        hrow['epoch'] = int(hrow['epoch'])
        hist.append(hrow)
    history.extend(hist)
    last = hist[-1] if hist else {}
    assert len(committed) == epochs
    if state:
        best = next(h for h in hist if h['epoch'] == state['best_epoch'])
        assert abs(best['val_macro_f1'] - state['best_val_macro_f1']) < 1e-10
        assert abs(max(h['val_macro_f1'] for h in hist) - state['best_val_macro_f1']) < 1e-10
    bdirs = sorted((d/'diagnostics/batches').glob('epoch-*'))
    latest_train = latest_val = None; train_count = val_count = 0
    for bd in bdirs:
        fs = sorted((bd/'train').glob('batch-*.json')); vs = sorted((bd/'validation').glob('batch-*.json'))
        train_count += len(fs); val_count += len(vs)
        if fs: latest_train = fs[-1]
        if vs: latest_val = vs[-1]
    b = read(latest_train) if latest_train else {}
    current_epoch = b.get('epoch') if epochs < planned and status.get('status') == 'running' else None
    partial_batches = b['batch'] + 1 if b and b['epoch'] > epochs else 0
    if latest_train: inventory_files.append(relative(latest_train))
    if latest_val: inventory_files.append(relative(latest_val))
    batch_total = math.ceil(f['train']/config['batch_size'])
    test = read(d/'test_metrics.json').get('classification', {})
    complete = status.get('status') == 'completed'
    if complete:
        assert epochs == planned and result['status'] == 'completed'
        assert abs(result['test_accuracy'] - test['accuracy']) < 1e-12
    r = {'dataset': ds, 'label': LABELS[ds], 'status': status.get('status','pending'),
         'stage': 'Concluído: treino, teste e estudo de ruído' if complete else ('Treinando' if status.get('status') == 'running' else 'Aguardando na fila'),
         'confirmedEpochs': epochs, 'plannedEpochs': planned, 'epochProgress': epochs/planned,
         'currentEpoch': current_epoch, 'currentSavedBatches': partial_batches, 'batchesPerEpoch': batch_total,
         'bestEpoch': state.get('best_epoch'), 'bestValMacroF1': state.get('best_val_macro_f1'),
         'lastTrainAccuracy': last.get('accuracy'), 'lastTrainLoss': last.get('loss'),
         'lastValAccuracy': last.get('val_accuracy'), 'lastValMacroF1': last.get('val_macro_f1'), 'lastValLoss': last.get('val_loss'),
         'testAccuracy': test.get('accuracy'), 'testMacroF1': test.get('macro_f1'),
         'durationHours': result.get('elapsed_this_session_seconds',0)/3600 if result else None,
         'epochSecondsTotal': sum(h['epoch_seconds'] for h in hist),
         'recentEpochSeconds': statistics.median(h['epoch_seconds'] for h in hist[-5:]) if hist else None,
         'completedAt': local(status.get('completed_at')),
         'currentBatchAccuracy': b.get('keras',{}).get('accuracy') if partial_batches else None,
         'currentBatchLoss': b.get('keras',{}).get('loss') if partial_batches else None,
         'latestBatchTime': datetime.fromtimestamp(latest_train.stat().st_mtime, TZ).isoformat() if latest_train else None,
         'optimizerSteps': state.get('optimizer_iterations',0),
         'classicalReferenceEligible': read(d/'classical_reference.json').get('eligible'),
         'classicalReferenceDifferences': ', '.join(read(d/'classical_reference.json').get('differences',[])),
         'runPath': relative(d), 'train': f['train'], 'val': f['val'], 'test': f['test'], 'classes': f['classes']}
    runs.append(r)
    if d.exists():
        inventory.append({'dataset': ds, 'label': LABELS[ds], 'confirmedEpochs': epochs,
          'trainBatchJson': train_count, 'validationBatchJson': val_count,
          'batchCircuitRecords': (train_count+val_count)*4,
          'savedKerasFiles': len(list((d/'checkpoints').glob('*.keras'))),
          'probeJson': len(list((d/'diagnostics/probes').glob('*.json'))),
          'testPredictions': (d/'test_predictions.npz').is_file(), 'noiseConditions': result.get('noise_study_conditions',0),
          'featuresHashFiles': len(config['source_feature_hashes'])})
        for rel in ('status.json','checkpoints/state.json','result.json','test_metrics.json','config.json','classical_reference.json'):
            if (d/rel).exists(): run_files.append(relative(d/rel))
        history_files.append(relative(d/'history.csv')); inventory_files.append(relative(d/'diagnostics/batches.csv'))
        tfs=sorted((d/'telemetry').glob('*/samples.jsonl'))
        for tf in tfs:
            first_t = last_t = None; count=0; peak_mem=0; peak_cpu=0
            with tf.open() as stream:
                for line in stream:
                    try: t=json.loads(line)
                    except json.JSONDecodeError: continue
                    first_t = first_t or t; last_t=t; count+=1
                    peak_mem=max(peak_mem,t.get('process',{}).get('tree_rss_mb',0) or 0)
                    peak_cpu=max(peak_cpu,t.get('process',{}).get('tree_cpu_percent',0) or 0)
            telemetry_files.append(relative(tf))
            if last_t:
                if not r['durationHours']: r['durationHours']=last_t['elapsed_seconds']/3600
                r['startedAt']=local(first_t['timestamp'])
                telemetry.append({'dataset': ds,'label': LABELS[ds], 'samples': count,
                  'startedAt': local(first_t['timestamp']), 'lastAt': local(last_t['timestamp']),
                  'elapsedHours': last_t['elapsed_seconds']/3600, 'lastCpuPercentTree': last_t.get('process',{}).get('tree_cpu_percent'),
                  'lastMemoryMbTree': last_t.get('process',{}).get('tree_rss_mb'), 'peakMemoryMbTree': peak_mem,
                  'peakCpuPercentTree': peak_cpu, 'freeDiskMb': last_t.get('disk',{}).get('run',{}).get('free_mb'),
                  'gpuAvailable': last_t.get('gpu',{}).get('available')})
        for p in (d/'diagnostics/noise-study').glob('*/conditions.csv'):
            noise_files.append(relative(p))
            cond=list(csv.DictReader(p.open()))
            assert len(cond) == result.get('noise_study_conditions')
            for c in cond:
                noise.append({'dataset':ds,'label':LABELS[ds],'condition':c['id'],
                  'noiseScale':float(c['noise_scale']), 'shots':int(c['shots']) if c['shots'] else None,
                  'samplingSeed':int(c['sampling_seed']) if c['sampling_seed'] else None,
                  **{k:float(c[k]) for k in ('accuracy','macro_f1','accuracy_degradation','fidelity_mean','purity_global_mean','entropy_global_bits_mean')}})
    checks.append({'dataset':ds,'historyMatchesCheckpoint':len(committed)==epochs,'testMatchesResult':True if complete else None})

grouped=defaultdict(list)
for n in noise:grouped[(n['dataset'],n['noiseScale'],n['shots'])].append(n)
noise_agg=[{'dataset':ds,'label':LABELS[ds], 'noiseScale':scale,'shots':shots,'shotsLabel':'Exato' if shots is None else str(shots),
            'conditions':len(ns), 'accuracy':statistics.mean(n['accuracy'] for n in ns),
            'minAccuracy':min(n['accuracy'] for n in ns),'maxAccuracy':max(n['accuracy'] for n in ns),
            'macroF1':statistics.mean(n['macro_f1'] for n in ns),
            'degradationPp':statistics.mean(n['accuracy_degradation'] for n in ns)*100}
           for (ds,scale,shots),ns in grouped.items()]
now=datetime.now(timezone.utc)
observed=now.isoformat()
overview={'totalRuns':len(runs),'completedRuns':sum(r['status']=='completed' for r in runs),
 'activeRuns':sum(r['status']=='running' for r in runs),'pendingRuns':sum(r['status']=='pending' for r in runs),
 'confirmedEpochs':sum(r['confirmedEpochs'] for r in runs),'plannedEpochs':sum(r['plannedEpochs'] for r in runs),
 'noiseConditions':len(noise),'telemetrySamples':sum(t['samples'] for t in telemetry),
 'trainBatchJson':sum(i['trainBatchJson'] for i in inventory),'validationBatchJson':sum(i['validationBatchJson'] for i in inventory),
 'batchCircuitRecords':sum(i['batchCircuitRecords'] for i in inventory),
 'testExamplesCompleted':sum(r['test'] for r in runs if r['status']=='completed'),
 'observedAt':now.astimezone(TZ).isoformat(), 'activeProcess':bool(active), 'processRows':active,
 'totalInputExamples':sum(f['total'] for f in features), 'featureWidth':128}
all_defs=[{'label':'Progresso por época','definition':'Épocas completas confirmadas em checkpoints/state.json divididas pelas 100 épocas planejadas de cada dataset. Não mede a fração do tempo total.','componentIds':['training-progress','run-table','report-summary']},
          {'label':'Acurácia de teste','definition':'Fração de acertos no conjunto completo de teste; modelo escolhido pelo maior macro-F1 de validação.','componentIds':['test-results','result-table','report-summary']},
          {'label':'Macro-F1','definition':'Média não ponderada dos F1 por classe; reportada separadamente para validação e teste.','componentIds':['result-table','training-curves','emnist-detail']}]
queries={
 'overview':{'rows':[overview],'source':source('Resumo da matriz Qiskit',[relative(ROOT/'matrix.json')]+run_files,all_defs,['Snapshot local; o treinamento continua após a leitura.'] )},
 'runs':{'rows':runs,'source':source('Etapas e resultados por dataset',run_files,all_defs,['Progresso parcial por lote não é checkpoint confirmado. Ausência de teste significa não coletado.'])},
 'features':{'rows':features,'source':source('Features128 e splits dos nove datasets',feature_files,[{'label':'Tamanho dos splits','definition':'Número de linhas dos arrays NPY já salvos; cada vetor tem 128 valores float32.','componentIds':['features-table','protocol']}],['Cabeçalhos conferidos no relatório; validação numérica completa executada pelo runner antes de gravar matrix.json.'])},
 'history':{'rows':history,'source':source('Histórico das épocas confirmadas',history_files,[{'label':'Curvas por época','definition':'Métricas de treino e validação somente das épocas já confirmadas.','componentIds':['training-curves','emnist-detail','generalization']}],['Resultados da última época não substituem o melhor checkpoint.'])},
 'inventory':{'rows':inventory,'source':source('Inventário de diagnósticos e arquivos',inventory_files,[{'label':'Registros por circuito','definition':'Cada JSON de lote contém quatro registros, um por circuito; inclui treino e validação.','componentIds':['inventory-table','collected-data']}],['Contagem de arquivos em snapshot durante execução; lotes da época corrente podem ser refeitos ao retomar.'])},
 'noise':{'rows':noise,'source':source('Condições do estudo de ruído',noise_files,[{'label':'Condições de ruído','definition':'Por dataset: quatro escalas (0, 0.5, 1, 2), uma condição exata e três números de shots com cinco seeds por escala: 64 condições.','componentIds':['noise-table','noise-chart','collected-data']}],['Estudo em simulação local sobre o checkpoint escolhido antes do teste.'])},
 'noise_aggregate':{'rows':noise_agg,'source':source('Média das condições com shots finitos',noise_files,[{'label':'Acurácia média com shots','definition':'Média aritmética das cinco seeds para uma combinação dataset, escala de ruído e número de shots; exato tem uma condição.','componentIds':['noise-table','noise-chart']}, {'label':'Queda em pontos percentuais','definition':'100 vezes acurácia ideal menos acurácia da condição, média entre seeds quando aplicável.','componentIds':['noise-table']}],['Variação min/max refere-se às cinco seeds, não a intervalo de confiança.'])},
 'telemetry':{'rows':telemetry,'source':source('Telemetria da execução',telemetry_files,[{'label':'Tempo de sessão','definition':'Elapsed_seconds da telemetria ou do resultado; inclui a etapa completa da execução quando concluída.','componentIds':['telemetry-table','emnist-detail']}],['CPU da árvore pode exceder 100% porque soma vários processos.'])}}
active_labels = [run['label'] for run in runs if run['status'] == 'running']
report_title = f"{overview['completedRuns']} treinamentos concluídos"
if active_labels:
    report_title += f"; {', '.join(active_labels)} segue em execução"
snapshot={'surface':'report','title':report_title,'status':'reviewed','buildStatus':'creating',
 'generatedAt':observed,'filters':[],'queries':queries,
 'report':{'asOf':observed,'timezone':'America/Sao_Paulo'},
 'collection':{'observedAt':observed,'validation':checks,'processRead':'ps -axo pid,ppid,etime,time,%cpu,%mem,state,command',
 'reproduction':'Execute src/content/report/collect_evidence.py com o Python da .venv e --output /caminho/novo-snapshot.json. O coletor é somente leitura.'}}
for q in queries.values():q['source']['queriedAt']=observed
Path(args.output).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'overview':overview,'runSummary':[{k:r[k] for k in ('dataset','status','confirmedEpochs','currentEpoch','currentSavedBatches','bestEpoch','testAccuracy','testMacroF1')} for r in runs], 'checks':checks},ensure_ascii=False))
