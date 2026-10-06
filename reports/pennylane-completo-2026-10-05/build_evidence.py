"""Read-only audit of the full PennyLane campaign; writes only this report folder.

Run with the Codex bundled Python (numpy/pandas), from any working directory.
No training, checkpoint mutation, or quantum re-simulation is performed.
"""
from pathlib import Path
import argparse, collections, hashlib, itertools, json, math, os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'outputs/quantum-parallel-128-20/runs'
OUT = HERE / 'evidence'
ORDER = ['mnist','fashion_mnist','kmnist','emnist_balanced','cifar10','cifar100_coarse','svhn','gtsrb','fer2013']
STAGES = ['embedding','layer_1','layer_2','layer_3']
NAMES = dict(zip(ORDER,['MNIST','Fashion-MNIST','KMNIST','EMNIST Balanced','CIFAR-10','CIFAR-100 coarse','SVHN','GTSRB','FER2013']))
QUERIES = {}
CATALOG = []
AUDIT = []

def read(p): return json.loads(p.read_text())
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x
def save(p, value): p.write_text(json.dumps(clean(value),ensure_ascii=False,separators=(',',':')))
def stats(a):
    a=np.asarray(a,dtype=float).reshape(-1); a=a[np.isfinite(a)]
    if not len(a): return dict(mean=None,min=None,max=None,std=None,count=0)
    return dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()),std=float(a.std()),count=len(a))
def add_query(q, rows, files, note, definitions=None):
    frame = pd.DataFrame(rows)
    if len(frame): frame.to_csv(OUT/f'{q}.csv',index=False,lineterminator='\n')
    QUERIES[q]={'rows':clean(rows),'source':{'label':q.replace('_',' '),'files':files,
        'executedAt':datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(),
        'caveats':[note], 'metricDefinitions':definitions or [],
        'evidenceFlow':[{'title':'Leitura local','detail':'; '.join(files)},
                        {'title':'Transformação reproduzível','detail':note+' Código: reports/pennylane-completo-2026-10-05/build_evidence.py'}]},
        'methods':[{'language':'python','code':'Executar build_evidence.py com numpy e pandas; função principal e blocos indicam todas as transformações.'}]}
def field_stats(a, prefix=''):
    if np.iscomplexobj(a):
        parts={'real':a.real,'imag':a.imag,'abs':np.abs(a)}
        return {prefix+k+'_'+s:v for k,b in parts.items() for s,v in stats(b).items()}
    return {prefix+s:v for s,v in stats(a).items()}

def main():
    OUT.mkdir(exist_ok=True)
    summary=[];history=[];classes=[];confusion=[];features=[];hardware=[];workers=[]
    hwstats=[];phases=[];batch_agg=[];batch_cost=[];probes=[];tensor_catalog=[]
    details=[];noise=[];noise_states=[];counts=[];prediction_stats=[];metadata=[];inventory=[]
    total_json=0
    for ds in ORDER:
        p=RUNS/ds/'pennylane/seed-42'; cfg=read(p/'config.json'); status=read(p/'status.json')
        print('READ',ds,flush=True)
        hist=pd.read_csv(p/'history.csv'); ft=read(p/'feature_statistics.json')
        tm=read(p/'test_metrics.json'); cl=tm['classification']; env=read(p/'environment.json')
        circuit=read(p/'circuit.json'); ref=read(p/'classical_reference.json'); profile=read(p/'profile.json')
        state=read(p/'checkpoints/state.json')
        assert status['status']=='completed' and list(hist.epoch)==list(range(1,101))
        selected=int(status['selected_epoch']); selected_row=hist.loc[hist.epoch==selected].iloc[0]
        assert abs(selected_row.val_macro_f1-status['selected_val_macro_f1'])<1e-10
        assert abs(cl['accuracy']-status['test_accuracy'])<1e-10
        assert abs(cl['macro_f1']-status['test_macro_f1'])<1e-10
        ntest=sum(v['support'] for v in cl['per_class'].values())
        assert ntest==ft['statistics']['test']['examples']==np.asarray(cl['confusion_matrix']).sum()
        record=dict(dataset=ds,name=NAMES[ds],epochs=len(hist),selected_epoch=selected,
                    train_examples=ft['statistics']['train']['examples'],val_examples=ft['statistics']['val']['examples'],
                    test_examples=ntest,num_classes=cfg['num_classes'],parameters=status['trainable_parameters'],
                    test_accuracy=cl['accuracy'],test_macro_f1=cl['macro_f1'],test_balanced_accuracy=cl['balanced_accuracy'],
                    test_macro_precision=cl['macro_precision'],test_macro_recall=cl['macro_recall'],test_auc=cl['macro_ovr_auc'],
                    test_loss=tm['keras']['loss'],val_macro_f1=status['selected_val_macro_f1'],
                    session_hours=status['elapsed_this_session_seconds']/3600,epoch_hours=hist.epoch_seconds.sum()/3600,
                    final_train_accuracy=hist.iloc[-1].accuracy,final_val_accuracy=hist.iloc[-1].val_accuracy,
                    gap_pp=100*(hist.iloc[-1].accuracy-hist.iloc[-1].val_accuracy),
                    completed_brt=datetime.fromisoformat(status['completed_at']).astimezone(ZoneInfo('America/Sao_Paulo')).isoformat(),
                    profile_examples_per_second=profile['examples_per_second'],profile_estimated_train_hours=profile['estimated_train_only_seconds']/3600,
                    classical_eligible=ref['eligible'],classical_differences=';'.join(ref['differences']))
        history.extend([dict(dataset=ds,**r) for r in hist.to_dict('records')])
        for label,v in cl['per_class'].items():
            names=ft['source_manifest']['features_manifest'].get('class_names',[])
            classes.append(dict(dataset=ds,class_id=int(label),class_name=names[int(label)] if len(names)>int(label) else label,**v))
        for i,row in enumerate(cl['confusion_matrix']):
            support=sum(row)
            for j,v in enumerate(row):confusion.append(dict(dataset=ds,true_class=str(i),predicted_class=str(j),count=v,row_rate=v/support if support else None))
        for split,v in ft['statistics'].items():features.append(dict(dataset=ds,split=split,**{k:a for k,a in v.items() if not isinstance(a,list)}))
        metadata.append(dict(dataset=ds,config=cfg,environment=env,circuit=circuit,profile=profile,
                    initialization=read(p/'initialization.json'),feature_statistics=ft,classical_reference=ref,
                    checkpoint_state=state,model_summary=(p/'model_summary.txt').read_text(),circuit_text=(p/'circuit.txt').read_text()))
        # All telemetry samples, including every child worker. MB fields are actually MiB in collector.
        tele_rows=[];sessions=[]
        for s in sorted((p/'telemetry').iterdir()):
            h=read(s/'hardware.json');srows=[]
            for line in (s/'samples.jsonl').open():
                d=json.loads(line);r=dict(dataset=ds,session=s.name,timestamp=d['timestamp'],elapsed_hours=d['elapsed_seconds']/3600,event=d['event'],phase=d.get('phase'))
                for family in ['system','process']:
                    for k,v in d[family].items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool) or v is None:r[family+'_'+k]=v
                for target,vals in d.get('disk',{}).items():
                    for k,v in vals.items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool):r['disk_'+target+'_'+k]=v
                r['gpu_available']=d.get('gpu',{}).get('available',False)
                srows.append(r)
                for pos,c in enumerate(d.get('process',{}).get('children',[])):
                    workers.append(dict(dataset=ds,session=s.name,elapsed_hours=r['elapsed_hours'],timestamp=d['timestamp'],worker=str(c['pid']),**c))
            tele_rows.extend(srows)
            sessions.append(dict(session=s.name,hardware=h,summary=read(s/'summary.json')))
        hardware.extend(tele_rows); tf=pd.DataFrame(tele_rows)
        record['telemetry_samples']=len(tf)
        for key in tf.select_dtypes(include='number').columns:
            hwstats.append(dict(dataset=ds,metric=key,missing=int(tf[key].isna().sum()),**stats(tf[key])))
        for (session,phase),g in tf.groupby(['session','phase'],dropna=False):
            phases.append(dict(dataset=ds,session=session,phase=None if pd.isna(phase) else phase,samples=len(g),
                first_hour=float(g.elapsed_hours.min()),last_hour=float(g.elapsed_hours.max())))
        record['cpu_system_mean']=float(tf.system_cpu_percent.mean()); record['cpu_tree_mean']=float(tf.process_tree_cpu_percent.mean())
        record['rss_tree_peak_gib']=float(tf.process_tree_rss_mb.max()/1024);record['rss_main_peak_gib']=float(tf.process_rss_mb.max()/1024)
        metadata[-1]['telemetry_sessions']=sessions
        # Read every batch JSON, not only the CSV: JSON additionally records parameters, throughput and Keras logs.
        digest=hashlib.sha256();json_count=0;csv_count=0;batch_bytes=0
        for ep in range(1,101):
            rows=[];cost=[]
            for phase in ['train','validation']:
                directory=p/'diagnostics/batches'/f'epoch-{ep:05d}'/phase
                for file in sorted(directory.glob('batch-*.json')):
                    raw=file.read_bytes();digest.update(file.name.encode());digest.update(raw);batch_bytes+=len(raw)
                    b=json.loads(raw);json_count+=1
                    cost.append(dict(dataset=ds,epoch=ep,phase=phase,batch_seconds=b['batch_seconds'],
                        evaluations=b['logical_circuit_evaluations'],examples_per_second=b['examples_per_second'],
                        evaluations_per_second=b['logical_evaluations_per_second'],
                        **{'keras_'+k:v for k,v in b['keras'].items() if isinstance(v,(int,float))}))
                    for br in b['circuits']:
                        row=dict(epoch=ep,phase=phase,circuit=br['circuit'],forward_seconds=br['forward_seconds'],backward_seconds=br['backward_seconds'],evaluations=br['evaluations'])
                        for family in ['inputs_radians','outputs_y','parameters','parameter_update']:
                            for k,v in br[family].items():row[family+'_'+k]=v
                        for gr in br['gradients']:
                            for k,v in gr.items():
                                if k not in ['evaluations','seconds']:row[k]=v
                        rows.append(row)
            f=pd.DataFrame(rows);csv_count+=len(f)
            for (phase,c),g in f.groupby(['phase','circuit']):
                row=dict(dataset=ds,epoch=ep,phase=phase,circuit=str(c),batches=len(g))
                for key in g.select_dtypes(include='number').columns:
                    if key in ['epoch','circuit']:continue
                    for stat,value in stats(g[key]).items():row[key+'_'+stat]=value
                    if key in ['forward_seconds','backward_seconds','evaluations']:row[key+'_sum']=float(g[key].sum())
                batch_agg.append(row)
            c=pd.DataFrame(cost)
            for phase,g in c.groupby('phase'):
                row=dict(dataset=ds,epoch=ep,phase=phase,batches=len(g),batch_seconds_sum=float(g.batch_seconds.sum()),evaluations_sum=int(g.evaluations.sum()))
                for key in g.select_dtypes(include='number').columns:
                    if key in ['epoch','evaluations']:continue
                    row.update({key+'_'+k:v for k,v in stats(g[key]).items()})
                batch_cost.append(row)
            if ep%25==0:print(ds,'batches',ep,flush=True)
        # Reconcile deterministic batch counts and public CSV line count.
        expected_train=100*math.ceil(record['train_examples']/cfg['batch_size']);expected_val=100*math.ceil(record['val_examples']/cfg['batch_size'])
        assert json_count==expected_train+expected_val and csv_count==4*json_count
        with (p/'diagnostics/batches.csv').open('rb') as stream:
            lines=sum(chunk.count(b'\n') for chunk in iter(lambda:stream.read(8*1024*1024),b''))-1
        assert lines==csv_count
        record.update(batch_files=json_count,circuit_records=csv_count,train_batches=expected_train,validation_batches=expected_val)
        inventory.append(dict(dataset=ds,category='batch_json',files=json_count,bytes=batch_bytes,pattern=str((p/'diagnostics/batches/epoch-*/{train,validation}/batch-*.json').relative_to(ROOT)),sha256_chain=digest.hexdigest()))
        total_json+=json_count
        # All probe tensors at epoch 0, all epochs, and selected checkpoint. Keep stage/branch identity.
        for file in sorted((p/'diagnostics/probes').glob('*.npz')):
            tag=file.stem;ep=selected if tag=='best' else int(tag.split('-')[1]);meta=read(file.with_suffix('.json'))
            with np.load(file,allow_pickle=False) as loaded: a={k:loaded[k] for k in loaded.files}
            for k in a:
                arr=a[k]
                tensor_catalog.append(dict(dataset=ds,family='ideal_probe',tag=tag,array=k,shape='×'.join(map(str,arr.shape)),dtype=str(arr.dtype),**field_stats(arr)))
            for branch in range(4):
                for stage,stage_name in enumerate(STAGES):
                    row=dict(dataset=ds,epoch=ep,tag=tag,circuit=str(branch),stage=stage_name,probes=len(a['indices']))
                    for k in a:
                        arr=a[k]
                        if arr.ndim>=3 and arr.shape[1:3]==(4,4):row.update(field_stats(arr[:,branch,stage],k+'_'))
                    # Angles and weights also retained per branch; radian raw values have no periodic wrapping.
                    row.update(field_stats(a['angles'][:,branch*5:(branch+1)*5],'angles_'))
                    row.update(field_stats(a['quantum_weights'][branch],'weights_'))
                    probes.append(row)
                    if tag=='best':
                        for k in a:
                            arr=a[k]
                            if arr.ndim<3 or arr.shape[1:3]!=(4,4):continue
                            mean=arr[:,branch,stage].mean(axis=0)
                            for coord in np.ndindex(mean.shape):
                                value=mean[coord]
                                details.append(dict(dataset=ds,circuit=str(branch),stage=stage_name,array=k,coordinate=','.join(map(str,coord)),
                                    value=float(value.real),imag=float(value.imag) if np.iscomplexobj(mean) else None,absolute=float(abs(value))))
            AUDIT.append(dict(dataset=ds,check='probability_normalization',tag=tag,error=meta['probability_sum_max_error'],ok=meta['probability_sum_max_error']<1e-9))
        # Every noise condition (including five measurement seeds); selected generation only.
        n=p/'diagnostics/noise-study';sel=read(n/'selection.json');g=n/sel['generation'];conditions=pd.read_csv(g/'conditions.csv')
        assert len(conditions)==64 and read(g/'summary.json')['status']=='complete'
        for r in conditions.to_dict('records'):
            r['dataset']=ds;r['shot_label']='exata' if pd.isna(r['shots']) else str(int(r['shots']))
            r['accuracy_change_pp']=-100*r['accuracy_degradation'];r['f1_change_pp']=-100*r['macro_f1_degradation'];noise.append(r)
        metadata[-1]['noise_summary']=read(g/'summary.json'); metadata[-1]['noise_selection']=sel
        for file in sorted((g/'probes').glob('*.npz')):
            scale=float(file.stem.split('-')[1])
            with np.load(file,allow_pickle=False) as loaded: a={k:loaded[k] for k in loaded.files}
            for k in a:
                arr=a[k]; tensor_catalog.append(dict(dataset=ds,family='noise_probe',tag=file.stem,array=k,shape='×'.join(map(str,arr.shape)),dtype=str(arr.dtype),**field_stats(arr)))
            for branch in range(4):
                for st,stage in enumerate(STAGES):
                    row=dict(dataset=ds,noise_scale=scale,circuit=str(branch),stage=stage)
                    for k in a:
                        arr=a[k]
                        if arr.ndim>=3 and arr.shape[1:3]==(4,4):row.update(field_stats(arr[:,branch,st],k+'_'))
                    noise_states.append(row)
        for file in sorted((g/'counts').glob('*.npz')):
            with np.load(file,allow_pickle=False) as loaded: a={k:loaded[k] for k in loaded.files}
            cond=next(r for r in noise if r['dataset']==ds and r['id']==file.stem)
            for k,arr in a.items():
                prediction_stats.append(dict(dataset=ds,file=str(file.relative_to(ROOT)),array=k,
                    shape='×'.join(map(str,arr.shape)),dtype=str(arr.dtype),**field_stats(arr)))
            for branch,basis,bit in itertools.product(range(4),range(3),range(32)):
                vals=a['counts_xyz'][:,branch,basis,bit]
                counts.append(dict(dataset=ds,condition=file.stem,noise_scale=cond['noise_scale'],shots=cond['shots'],sampling_seed=cond['sampling_seed'],
                    circuit=str(branch),basis='XYZ'[basis],bitstring=f'{bit:05b}',counts=int(vals.sum()),examples=len(vals),
                    probability=float(vals.mean()/cond['shots']),probe_counts=int(a['probe_counts_xyz'][:,branch,basis,bit].sum()),
                    standard_error_mean=float(a['standard_error_xyz'][:,branch,basis].mean())))
            assert np.all(a['counts_xyz'].sum(axis=-1)==int(cond['shots']))
        # Prediction arrays: all conditions and ideal test; summarize every saved field.
        predpaths=[p/'test_predictions.npz',g/'ideal-test.npz',*sorted((g/'predictions').glob('*.npz')),
                   *sorted(g.glob('noise-*-test-probabilities.npz')),p/'initial_weights.npz']
        for file in predpaths:
            a=np.load(file,allow_pickle=False)
            for k in a.files:
                arr=a[k];prediction_stats.append(dict(dataset=ds,file=str(file.relative_to(ROOT)),array=k,shape='×'.join(map(str,arr.shape)),dtype=str(arr.dtype),**field_stats(arr)))
            a.close()
        # Compact category inventory covers all raw files without copying the large campaign.
        cats=collections.defaultdict(lambda:[0,0])
        for directory,dirs,files in os.walk(p):
            rel=Path(directory).relative_to(p)
            if str(rel).startswith('diagnostics/batches'):dirs[:]=[];continue
            for name in files:
                file=Path(directory)/name;parts=file.relative_to(p).parts
                category='/'.join(parts[:3]) if parts[0]=='diagnostics' and len(parts)>3 else ('/'.join(parts[:-1]) or 'run_metadata')
                cats[category][0]+=1;cats[category][1]+=file.stat().st_size
        for cat,(count,size) in cats.items():inventory.append(dict(dataset=ds,category=cat,files=count,bytes=size,pattern=str(p.relative_to(ROOT))+'/'+cat))
        summary.append(record);print('DONE',ds,'JSON',json_count,flush=True)
    wildcard='outputs/quantum-parallel-128-20/runs/<dataset>/pennylane/seed-42/'
    add_query('runs',summary,[wildcard+'{status,config,test_metrics}.json',wildcard+'history.csv'],
        'Somente nove runs full concluídas, seed 42, 100 épocas; teste selecionado por val_macro_f1. Sem comparação clássica elegível.')
    add_query('history',history,[wildcard+'history.csv'],'Todas as linhas e colunas originais; 900 épocas. Métricas por classe são validação, não teste.')
    add_query('classes',classes,[wildcard+'test_metrics.json'],'Uma linha por classe no teste completo, no checkpoint selecionado.')
    add_query('confusion',confusion,[wildcard+'test_metrics.json'],'Contagens completas, incluindo zeros observados; taxa por linha = contagem / suporte da classe real.')
    add_query('features',features,[wildcard+'feature_statistics.json'],'Estatísticas coletadas das features128 em treino, validação e teste; não são imagens brutas.')
    add_query('hardware',hardware,[wildcard+'telemetry/<sessão>/samples.jsonl'],'Todas as amostras originais, sem downsampling no conjunto de dados. Campos *_mb do coletor são MiB (divisão por 1024²); CPU de processo/árvore pode exceder 100%; sistema varia de 0 a 100%.')
    add_query('workers',workers,[wildcard+'telemetry/<sessão>/samples.jsonl'],'Todos os processos filhos em cada amostra; árvore soma RSS sem descontar páginas compartilhadas.')
    add_query('hardware_stats',hwstats,[wildcard+'telemetry/<sessão>/samples.jsonl'],'Média aritmética, mínimo, máximo, desvio populacional, contagem não nula e faltantes por campo. Não são médias ponderadas por tempo.')
    add_query('phases',phases,[wildcard+'telemetry/<sessão>/samples.jsonl'],'Somente amostras com phase explícita; o coletor não mantém phase nas amostras de intervalo, portanto não permite atribuir toda a sessão às fases.')
    add_query('batch_metrics',batch_agg,[wildcard+'diagnostics/batches/epoch-*/<fase>/batch-*.json'],
        'Leitura de TODOS os JSONs: por época/fase/circuito, média/min/max/std/count de cada estatística coletada, soma dos tempos por worker e avaliações. Média das variâncias de batch não é variância global de exemplos. Tempos dos quatro ramos se sobrepõem.')
    add_query('batch_cost',batch_cost,[wildcard+'diagnostics/batches/epoch-*/<fase>/batch-*.json'],
        'Um registro por batch físico, agregado por época/fase. batch_seconds somado uma vez, não quatro. Keras em batches são logs cumulativos; sua média não substitui o histórico de época.')
    add_query('probes',probes,[wildcard+'diagnostics/probes/{epoch-00000..epoch-00100,best}.npz'],
        'Todas as 102 sondas por run, resumindo todos os tensores por ramo/estágio. Duas amostras fixas de validação por classe; média não representa todo o dataset. Complexos: parte real, imaginária e módulo separados.')
    add_query('probe_details',details,[wildcard+'diagnostics/probes/best.npz'],
        'Cada coordenada dos tensores do checkpoint selecionado, com média apenas sobre as sondas fixas; coordenadas zero-based. Inclui estados, densidades reduzidas, Bloch, probabilidades e correlações XYZ.')
    add_query('noise',noise,[wildcard+'diagnostics/noise-study/<geração selecionada>/conditions.csv'],
        'Todas as 576 condições. Cinco seeds de amostragem por shots finitos; dispersão entre seeds não é intervalo de confiança de treinamento. Degradação registrada = baseline - observado; mudança_pp = -100 × degradação. SNR null com zero_error_infinite significa +∞.')
    add_query('noise_states',noise_states,[wildcard+'diagnostics/noise-study/<geração selecionada>/probes/noise-*.npz'],
        'Todos os tensores das quatro escalas de ruído por estágio/ramo. Estado antes da medição; shots não alteram esses estados. Entropia reduzida em estados mistos não é medida isolada de emaranhamento.')
    add_query('counts',counts,[wildcard+'diagnostics/noise-study/<geração selecionada>/counts/*.npz'],
        'Todos os 540 arquivos de counts: soma sobre exemplos por condição/ramo/base/bitstring; probabilidade = counts/(exemplos×shots). Bases XYZ amostradas independentemente. Soma de counts conferida igual a shots para todos os exemplos/ramo/base.')
    add_query('tensor_catalog',tensor_catalog,[wildcard+'diagnostics/{probes,noise-study/<geração>/probes}/*.npz'],
        'Catálogo de cada array efetivamente lido, shape, dtype e resumo; valores complexos separados em real/imag/abs. Vetores e matrizes completos permanecem em NPZ originais.')
    add_query('prediction_stats',prediction_stats,[wildcard+'{test_predictions,initial_weights}.npz',wildcard+'diagnostics/noise-study/<geração>/*.npz',wildcard+'diagnostics/noise-study/<geração>/counts/*.npz'],
        'Estatísticas de cada campo dos arquivos de predições, probabilidades de ruído, pesos iniciais e counts, incluindo leituras XYZ/erro padrão das sondas; identificadores e labels são metadados, não medidas contínuas de qualidade.')
    add_query('inventory',inventory,[wildcard],'Inventário por categoria, número de arquivos e bytes. SHA256 encadeado dos bytes de JSONs de batch em ordem determinística; não é hash de um único arquivo.')
    save(OUT/'metadata.json',metadata);save(OUT/'checks.json',AUDIT)
    snapshot=read(HERE/'app/src/data.json');snapshot.update(title='Cabeça quântica PennyLane: os nove treinamentos e todos os diagnósticos',surface='report',status='observed',buildStatus='creating',
        generatedAt=datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(),report={'asOf':'2026-10-05','visibleFilterIds':[]},filters=[],queries=QUERIES)
    snapshot['campaign']={'runs':len(summary),'epochs':len(history),'batch_files':total_json,'circuit_records':sum(r['circuit_records'] for r in summary),
                         'noise_conditions':len(noise),'telemetry_samples':len(hardware),'test_examples':sum(r['test_examples'] for r in summary),
                         'session_hours':sum(r['session_hours'] for r in summary),'metadata':metadata,'checks':{'probability_checks':len(AUDIT),'failed':sum(not r['ok'] for r in AUDIT)}}
    save(HERE/'app/src/data.json',snapshot);save(HERE/'reviewed-snapshot.json',snapshot)
    print('COMPLETE',snapshot['campaign']|{'metadata':'see evidence/metadata.json'},flush=True)

if __name__=='__main__':main()
