"""Publication-ready scientific figures from the audited CSVs (no simulation)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
E=HERE/'evidence';F=HERE/'figures';F.mkdir(exist_ok=True)
NAMES={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST','emnist_balanced':'EMNIST Balanced','cifar10':'CIFAR-10','cifar100_coarse':'CIFAR-100 coarse','svhn':'SVHN','gtsrb':'GTSRB','fer2013':'FER2013'}
COLORS=['#3569a8','#bd9135','#d17441','#758444','#b3698c']
NOISE_LABELS={'accuracy':('Acurácia','Fração 0–1'),'macro_f1':('Macro-F1','Fração 0–1'),
 'accuracy_degradation':('Degradação da acurácia','Baseline − observado (fração)'),
 'macro_f1_degradation':('Degradação do Macro-F1','Baseline − observado (fração)'),
 'accuracy_change_pp':('Mudança da acurácia','Pontos percentuais'),
 'f1_change_pp':('Mudança do Macro-F1','Pontos percentuais'),
 'entropy_global_bits_mean':('Entropia global média','Bits'),'fidelity_mean':('Fidelidade média','Fidelidade'),
 'fidelity_min':('Fidelidade mínima','Fidelidade'),'output_mae':('MAE das saídas Y','MAE'),
 'output_rmse':('RMSE das saídas Y','RMSE'),'purity_global_mean':('Pureza global média','Tr(ρ²)'),
 'relative_l2_error':('Erro L2 relativo das saídas','Erro relativo'),'shot_standard_error_y_mean':('Erro padrão médio de Y','Erro padrão'),
 'snr_db_auxiliary':('SNR auxiliar das saídas Y','dB'),'trace_distance_max':('Distância de traço máxima','Distância de traço'),
 'trace_distance_mean':('Distância de traço média','Distância de traço')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
CAT=[]
def save(fig,name,title,note,dataset='all'):
    fig.suptitle(title,fontsize=13,ha='left',x=.09,y=.98)
    fig.text(.09,.012,note,fontsize=8,ha='left',va='bottom',wrap=True)
    fig.tight_layout(rect=(0,.07,1,.94))
    for ext in ['png','svg']:fig.savefig(F/f'{name}.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig);CAT.append(dict(file=name,title=title,note=note,dataset=dataset))
def line(frame,x,fields,labels,title,name,unit,note,ds,scale=1):
    fig,ax=plt.subplots(figsize=(10,4.4))
    for i,k in enumerate(fields):ax.plot(frame[x],frame[k]*scale,label=labels[i],color=COLORS[i%5],linestyle=['-','--',':','-.'][i%4],linewidth=1.4)
    ax.set(xlabel='Época' if x=='epoch' else 'Tempo decorrido (h)',ylabel=unit);ax.legend(loc='best')
    save(fig,name,title,note,ds)
def circuit_lines(frame,metric,title,name,unit,note,ds):
    fig,ax=plt.subplots(figsize=(10,4.4))
    for i,(c,g) in enumerate(frame.groupby('circuit')):ax.plot(g.epoch,g[metric],label=f'Circuito {c}',color=COLORS[i],linestyle=['-','--',':','-.'][i],linewidth=1.4)
    ax.set(xlabel='Época',ylabel=unit);ax.legend(loc='best');save(fig,name,title,note,ds)
def main():
    runs=pd.read_csv(E/'runs.csv');hist=pd.read_csv(E/'history.csv');hw=pd.read_csv(E/'hardware.csv');batch=pd.read_csv(E/'batch_metrics.csv');probes=pd.read_csv(E/'probes.csv');noise=pd.read_csv(E/'noise.csv');classes=pd.read_csv(E/'classes.csv');confusion=pd.read_csv(E/'confusion.csv');details=pd.read_csv(E/'probe_details.csv');counts=pd.read_csv(E/'counts.csv');noisystates=pd.read_csv(E/'noise_states.csv')
    fig,ax=plt.subplots(figsize=(10,5.6));y=np.arange(len(runs));ax.barh(y-.17,100*runs.test_accuracy,.32,label='Acurácia',color=COLORS[0]);ax.barh(y+.17,100*runs.test_macro_f1,.32,label='Macro-F1',color=COLORS[1],hatch='//');ax.set(yticks=y,yticklabels=runs.name,xlim=(0,100),xlabel='Teste (%)');ax.invert_yaxis();ax.legend();save(fig,'01-resultados','Resultados dos nove datasets','Checkpoint escolhido por Macro-F1 de validação; teste completo. Datasets têm classes e dificuldades distintas.')
    fig,axs=plt.subplots(1,2,figsize=(12,5));axs[0].barh(y,runs.session_hours,color=COLORS[0]);axs[0].set(yticks=y,yticklabels=runs.name,xlabel='Sessão total (h)');axs[1].barh(y,runs.rss_tree_peak_gib,color=COLORS[0],label='Árvore');axs[1].barh(y,runs.rss_main_peak_gib,color=COLORS[1],label='Principal');axs[1].set(yticks=y,yticklabels=runs.name,xlabel='Pico RSS (GiB)');axs[1].legend();[a.invert_yaxis() for a in axs];save(fig,'02-hardware-tempo','Tempo e memória por dataset','RSS amostrado; soma de RSS dos processos pode incluir páginas compartilhadas.')
    metrics=[('weight_gradient_norm_mean','Norma dos gradientes de pesos','Norma L2'),('weight_gradient_variance_mean','Variância dos gradientes de pesos','Variância'),('input_gradient_norm_mean','Norma dos gradientes de entrada','Norma L2'),('per_example_gradient_variance_mean_mean','Variância dos gradientes por exemplo','Variância'),('parameter_update_norm_mean','Norma da atualização dos ângulos','Norma L2 (rad)'),('inputs_radians_mean_mean','Ângulos de entrada','Radianos'),('outputs_y_mean_mean','Expectativas Pauli Y','Expectativa Y'),('parameters_mean_mean','Pesos quânticos','Radianos'),('forward_seconds_mean','Tempo de forward por circuito','Segundos/batch'),('backward_seconds_mean','Tempo de backward por circuito','Segundos/batch')]
    pm=[('entropy_global_bits_mean','Entropia global','Bits'),('entropy_one_qubit_bits_mean','Entropia de 1 qubit','Bits'),('entropy_two_qubit_bits_mean','Entropia de 2 qubits','Bits'),('measurement_entropy_z_bits_mean','Entropia de medição Z','Bits'),('purity_global_mean','Pureza global','Tr(ρ²)'),('purity_one_qubit_mean','Pureza de 1 qubit','Tr(ρ²)'),('purity_two_qubit_mean','Pureza de 2 qubits','Tr(ρ²)'),('pauli_y_variance_mean','Variância de Pauli Y','Variância'),('yy_correlations_mean','Correlação YY','Correlação'),('yy_covariance_mean','Covariância YY','Covariância'),('bloch_xyz_mean','Média das componentes Bloch XYZ','Componente média')]
    for ds,name in NAMES.items():
        print('PLOT',ds,flush=True);h=hist[hist.dataset==ds];b=batch[(batch.dataset==ds)&(batch.phase=='train')];p=probes[(probes.dataset==ds)&(probes.tag!='best')&(probes.stage=='layer_3')];w=hw[hw.dataset==ds];n=noise[noise.dataset==ds];cl=classes[classes.dataset==ds]
        line(h,'epoch',['loss','val_loss'],['Treino','Validação'],name+' — loss',ds+'-loss','Loss','100 épocas; nenhuma seleção pelo teste.',ds)
        line(h,'epoch',['accuracy','val_accuracy','val_macro_f1'],['Acurácia treino','Acurácia validação','Macro-F1 validação'],name+' — aprendizado',ds+'-aprendizado','Qualidade (%)','A melhor época é selecionada por Macro-F1 de validação.',ds,100)
        line(h,'epoch',['epoch_seconds'],['Tempo'],name+' — tempo por época',ds+'-tempo','Segundos','Tempo do callback por época; não isola o tempo de simulação.',ds)
        line(h,'epoch',['train_examples_per_second'],['Throughput'],name+' — throughput por época',ds+'-throughput','Exemplos/s','Inclui o intervalo medido pelo callback de época.',ds)
        line(w,'elapsed_hours',['system_cpu_percent','process_cpu_percent','process_tree_cpu_percent'],['Sistema','Principal','Árvore'],name+' — CPU',ds+'-cpu','CPU (%)','Todas as amostras. Sistema 0–100%; árvore soma utilização por core e pode exceder 100%.',ds)
        memory=w.copy();memory[['process_rss_mb','process_tree_rss_mb']]/=1024
        line(memory,'elapsed_hours',['process_rss_mb','process_tree_rss_mb'],['Principal','Árvore'],name+' — RSS',ds+'-rss','GiB','Todas as amostras; soma RSS pode incluir páginas compartilhadas.',ds)
        line(w,'elapsed_hours',['system_memory_percent'],['RAM'],name+' — memória do sistema',ds+'-ram','RAM (%)','Memória unificada do sistema; não é exclusivamente atribuível ao treinamento.',ds)
        line(w,'elapsed_hours',['disk_run_free_mb'],['Espaço livre'],name+' — disco livre',ds+'-disco','MiB livres','Espaço livre no filesystem da run; não mede I/O nem escrita exclusiva deste processo.',ds)
        for key,title,unit in metrics:circuit_lines(b,key,name+' — '+title,ds+'-'+key,unit,'Média das estatísticas de TODOS os batches de treino em cada época e circuito.',ds)
        for key,title,unit in pm:circuit_lines(p,key,name+' — '+title,ds+'-'+key,unit,'Sondas fixas de validação; estágio layer_3. Não representam todo o dataset.',ds)
        # All validation classes, all four metrics, across all epochs.
        for metric in ['f1','precision','recall','support']:
            values=[]
            for k in sorted(int(c) for c in cl.class_id):values.append(h[f'val_class_{k}_{metric}'].to_numpy())
            fig,ax=plt.subplots(figsize=(10,max(4,len(values)*.15)));im=ax.imshow(np.array(values),aspect='auto',cmap='Blues',vmin=0 if metric!='support' else None,vmax=1 if metric!='support' else None,extent=[.5,100.5,len(values)-.5,-.5]);ax.set(xlabel='Época',ylabel='ID da classe',yticks=np.arange(len(values)));fig.colorbar(im,ax=ax,label=metric+' (fração)' if metric!='support' else 'Exemplos');save(fig,ds+'-classes-val-'+metric,name+' — '+metric+' por classe na validação','Todas as classes e todas as épocas. Valores de validação, sem usar teste para seleção.',ds)
        fig,ax=plt.subplots(figsize=(10,max(4,len(cl)*.22)));y=np.arange(len(cl));ax.barh(y-.2,100*cl.precision,.2,color=COLORS[0],label='Precisão');ax.barh(y,100*cl.recall,.2,color=COLORS[1],hatch='//',label='Recall');ax.barh(y+.2,100*cl.f1,.2,color=COLORS[2],hatch='..',label='F1');ax.set(yticks=y,yticklabels=cl.class_name,xlim=(0,100),xlabel='Teste (%)');ax.invert_yaxis();ax.legend();save(fig,ds+'-classes-teste',name+' — métricas de teste por classe','Teste completo, checkpoint selecionado; suporte disponível no CSV de classes.',ds)
        cm=confusion[confusion.dataset==ds].pivot(index='true_class',columns='predicted_class',values='count').sort_index().sort_index(axis=1)
        fig,ax=plt.subplots(figsize=(max(6,len(cm)*.17),max(5,len(cm)*.17)));im=ax.imshow(cm,cmap='Blues',vmin=0);ax.set(xlabel='Classe predita',ylabel='Classe real',xticks=np.arange(len(cm)),yticks=np.arange(len(cm)));fig.colorbar(im,ax=ax,label='Exemplos');save(fig,ds+'-confusao',name+' — matriz de confusão','Todas as células, inclusive zeros; contagens do teste completo.',ds)
        # Every numeric noise metric except structural identifiers; panels compare all shots and seeds.
        noise_keys=[k for k in n.select_dtypes(include='number') if k not in ['sampling_seed','noise_scale','shots','test_examples']]
        for key in noise_keys:
            fig,ax=plt.subplots(figsize=(10,4.4))
            for i,shots in enumerate(['exata','256','1024','4096']):
                g=n[n.shot_label.astype(str)==shots].groupby('noise_scale')[key].agg(['mean','min','max']).reset_index()
                if g['mean'].notna().any():
                    ax.plot(g.noise_scale,g['mean'],label=shots,color=COLORS[i],linestyle=['-','--',':','-.'][i]);ax.fill_between(g.noise_scale,g['min'],g['max'],color=COLORS[i],alpha=.1)
            title,unit=NOISE_LABELS.get(key,(key,key))
            ax.set(xlabel='Escala de ruído',ylabel=unit);ax.legend(title='Shots')
            save(fig,ds+'-ruido-'+key,name+' — '+title,'Linha: média das cinco seeds de shots; faixa: mínimo–máximo, não intervalo de confiança. Exata: uma condição.',ds)
        # Best checkpoint: every tensor coordinate, real/imag/absolute for complex data.
        d=details[(details.dataset==ds)&(details.stage=='layer_3')]
        for arr,g in d.groupby('array'):
            fig,ax=plt.subplots(figsize=(10,4.4))
            for i,(branch,branchrows) in enumerate(g.groupby('circuit')):
                coords=[tuple(int(x) for x in c.split(',')) if str(c)!='' else () for c in branchrows.coordinate.fillna('')]
                order=sorted(range(len(coords)),key=lambda i:coords[i]);branchrows=branchrows.iloc[order]
                ax.plot(np.arange(len(branchrows)),branchrows.value,label=f'C{branch} real',color=COLORS[i],linestyle=['-','--',':','-.'][i],linewidth=1)
                if branchrows.imag.notna().any():ax.plot(np.arange(len(branchrows)),branchrows.imag,label=f'C{branch} imaginário',color=COLORS[i],linestyle=':',alpha=.5)
            ax.set(xlabel='Coordenadas em ordem lexicográfica (zero-based)',ylabel=arr);ax.legend(ncol=2)
            save(fig,ds+'-tensor-'+arr,name+' — '+arr,'Média sobre sondas fixas, melhor checkpoint, layer_3; coordenadas completas disponíveis no CSV.',ds)
        # Histograms of counts over all test examples, selected representative condition.
        cc=counts[(counts.dataset==ds)&(counts.noise_scale==0)&(counts.shots==256)&(counts.sampling_seed==42)&(counts.circuit==0)]
        fig,ax=plt.subplots(figsize=(11,4.4));xx=np.arange(32)
        for i,basis in enumerate('XYZ'):
            g=cc[cc.basis==basis].sort_values('bitstring');ax.bar(xx+(i-1)*.23,g.probability,.23,label=basis,color=COLORS[i],hatch=['','//','..'][i])
        ax.set(xticks=xx,xticklabels=[f'{x:05b}' for x in xx],ylabel='Frequência relativa',xlabel='Bitstring; q0 à esquerda');ax.tick_params(axis='x',rotation=60);ax.legend(title='Base');save(fig,ds+'-counts',name+' — counts XYZ','Escala 0, 256 shots, seed 42, circuito 0. As demais 59 condições finitas estão no explorador e CSV.',ds)
    (F/'catalog.json').write_text(json.dumps(CAT,ensure_ascii=False,indent=2))
    print('FIGURES',len(CAT),flush=True)
if __name__=='__main__':main()
