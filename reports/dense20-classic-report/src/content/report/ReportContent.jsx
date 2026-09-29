import React from "react";

import {
  DataComponent,
  DataTable,
  EvidenceChart,
  MetricCard,
  ReportSection,
  RichNarrative,
  useDataApp,
} from "../../data-app-public.jsx";

const score = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
const percent = (value, digits = 2) => Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const points = (value) => Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(3)} p.p.` : "—";
const integer = (value) => Number.isFinite(value) ? new Intl.NumberFormat("pt-BR").format(value) : "—";
const minutes = (value) => Number.isFinite(value) ? `${(value / 60).toFixed(1)} min` : "—";

const deltaSpec = {
  type: "bar",
  x: "dataset",
  y: "changePp",
  xLabel: "Dataset",
  yLabel: "Mudança no Macro-F1 (p.p.)",
  valueDecimals: 3,
};

const costSpec = {
  type: "bar",
  x: "dataset",
  y: "trainingMinutes",
  xLabel: "Dataset",
  yLabel: "Tempo das 100 épocas (min)",
  valueDecimals: 1,
};

export function ReportContent() {
  const {
    reviewedRows,
    chartOverrides,
    visible,
    canEdit,
    mode,
    appTitle,
    setAppTitle,
  } = useDataApp();

  const results = reviewedRows("dense20_results");
  const summaryRows = reviewedRows("experiment_summary");
  const features128Rows = reviewedRows("features128_exports");
  const diagnostics = reviewedRows("class_diagnostics");
  const summary = summaryRows[0] ?? {};
  const features128 = features128Rows[0] ?? {};
  const sortedDeltaRows = results
    .map(({ dataset, changePp }) => ({ dataset, changePp }))
    .sort((left, right) => right.changePp - left.changePp);
  const costRows = results
    .map(({ dataset, trainingSeconds }) => ({ dataset, trainingMinutes: trainingSeconds / 60 }))
    .sort((left, right) => right.trainingMinutes - left.trainingMinutes);
  const deltaChart = chartOverrides["report-macro-f1-comparison"] ?? deltaSpec;
  const costChart = chartOverrides["report-training-cost"] ?? costSpec;
  const evidence = {
    dense20_results: results,
    experiment_summary: summaryRows,
    features128_exports: features128Rows,
  };

  return <article className="report-content" aria-label="Relatório técnico da rodada Dense20">
    <header className="report-hero">
      <p className="report-kicker">Rodada clássica · backbone congelado · seed 42</p>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "Editar título do relatório" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:description" className="report-deck" label="Editar introdução"
        value="Relatório auditável das nove execuções clássicas Dense20 e da exportação corrigida da interface reutilizável de **128 dimensões**, obtida após o pooling e antes de qualquer camada densa." />
    </header>

    {visible("report-executive-summary") && <ReportSection id="report-executive-summary"
      title="Síntese executiva" queryId="experiment_summary"
      queryIds={["experiment_summary", "dense20_results"]} sourceRowsByQuery={evidence}
      sourceRows={summaryRows} showHeading={false} className="report-summary">
      <RichNarrative id="report-executive-summary:body" className="report-summary-lead" label="Editar síntese"
        value={`## O gargalo de 20 neurônios preservou o desempenho e melhorou a maioria das bases

As nove execuções terminaram com sucesso. O Macro-F1 médio não ponderado passou de **${score(summary.meanOriginalMacroF1)}** para **${score(summary.meanDense20MacroF1)}**, uma variação observada de **${points(summary.meanChangePp)}**. Houve melhora em **${summary.improved} de ${summary.datasets} datasets**; as duas quedas foram menores que 0,08 ponto percentual.

O maior ganho ocorreu no **${summary.largestGainDataset}** (${points(summary.largestGainPp)}), seguido por SVHN (+0,733 p.p.). A maior retração foi no **${summary.largestRegressionDataset}** (${points(summary.largestRegressionPp)}). Todos os fingerprints de split coincidiram e todos os hashes do backbone permaneceram inalterados.

**Leitura correta:** estes números medem a cabeça Dense20 completa. O vetor de 20 dimensões é uma saída treinada dessa cabeça, não a entrada para substituir toda a ponta densa. Para novos treinamentos de cabeça, a interface correta é o vetor N×128 exportado antes de Dense(20). Os resultados também não provam superioridade estatística: cada dataset foi executado com uma única seed e não há intervalo de confiança.`} />
    </ReportSection>}

    <div className="report-facts" aria-label="Indicadores principais">
      {visible("report-metric-completed") && <MetricCard id="report-metric-completed" title="Execuções concluídas"
        queryId="experiment_summary" sourceRows={summaryRows} value={`${summary.completed}/${summary.datasets}`}
        comparison="100 épocas cada" deltaTone="neutral"
        description="Runs com status completed e artefatos auditados." />}
      {visible("report-metric-improved") && <MetricCard id="report-metric-improved" title="Datasets com ganho"
        queryId="experiment_summary" sourceRows={summaryRows} value={`${summary.improved}/${summary.datasets}`}
        comparison={`${points(summary.meanChangePp)} na média`} deltaTone="positive"
        description="Sinal positivo significa Macro-F1 Dense20 acima do checkpoint original." />}
      {visible("report-metric-backbone") && <MetricCard id="report-metric-backbone" title="Backbones alterados"
        queryId="experiment_summary" sourceRows={summaryRows} value="0"
        comparison="hash antes = depois" deltaTone="positive"
        description="Nenhuma variável do backbone foi treinável ou modificada." />}
      {visible("report-metric-features") && <MetricCard id="report-metric-features" title="Vetores exportados"
        queryId="features128_exports" sourceRows={features128Rows} value={integer(features128.totalRows)}
        comparison={`${features128.featureWidth} dimensões`} deltaTone="neutral"
        description="Saída de block5_pool + GAP, antes de qualquer Dense; amostras originais em ordem determinística." />}
    </div>

    {visible("report-macro-f1-comparison") && <section className="report-section">
      <ReportSection id="report-results" title="Resultados por dataset" queryId="dense20_results"
        sourceRows={results} showHeading={false}>
        <RichNarrative id="report-results:body" className="report-analysis" label="Editar análise dos resultados"
          value={`## Resultados por dataset

Os ganhos se concentraram nas bases em que a cabeça original ainda deixava margem para reorganizar a representação congelada: **EMNIST Balanced (+1,124 p.p.)** e **SVHN (+0,733 p.p.)**. MNIST, GTSRB, CIFAR-100 coarse, KMNIST e FER2013 também avançaram.

Fashion-MNIST e CIFAR-10 recuaram apenas **0,080** e **0,078 p.p.**, respectivamente. Essas diferenças são pequenas demais para uma conclusão causal com uma única seed. Em CIFAR-10, a acurácia até subiu levemente enquanto o Macro-F1 caiu, indicando redistribuição de desempenho entre classes, não degradação uniforme.

No gráfico, valores positivos favorecem Dense20. Isso normaliza a leitura em relação ao campo histórico \`delta_macro_f1\` dos artefatos, que foi definido no protocolo com o sinal oposto: original menos Dense20.`} />
      </ReportSection>
      <EvidenceChart id="report-macro-f1-comparison" queryId="dense20_results"
        title="Variação do Macro-F1 após a cabeça Dense20"
        description="Pontos percentuais: Macro-F1 Dense20 menos Macro-F1 original."
        spec={deltaChart} rows={sortedDeltaRows} sourceRows={results} height={340} />
    </section>}

    {visible("report-results-table") && <DataComponent id="report-results-table" title="Comparação completa dos nove datasets"
      queryId="dense20_results" kind="table" displayRows={results} sourceRows={results}
      description="Métricas do checkpoint selecionado por Macro-F1 de validação e avaliado uma única vez no teste.">
      <DataTable rows={results} rowKey="datasetId" caption="Resultados Dense20 por dataset" columns={[
        { field: "dataset", label: "Dataset", presentation: "identity" },
        { field: "originalMacroF1", label: "Macro-F1 original", renderCell: score },
        { field: "dense20MacroF1", label: "Macro-F1 Dense20", renderCell: score },
        { field: "changePp", label: "Mudança", renderCell: points,
          deltaTone: (value) => value > 0 ? "positive" : value < 0 ? "negative" : "neutral" },
        { field: "dense20Accuracy", label: "Acurácia", renderCell: percent },
        { field: "balancedAccuracy", label: "Acurácia balanceada", renderCell: percent },
        { field: "macroAuc", label: "Macro-AUC", renderCell: score },
        { field: "bestEpoch", label: "Melhor época" },
        { field: "trainingSeconds", label: "Tempo", renderCell: minutes },
      ]} />
    </DataComponent>}

    {visible("report-class-diagnostics") && <section className="report-section">
      <ReportSection id="report-errors" title="Onde os modelos ainda erram" queryId="class_diagnostics"
        sourceRows={diagnostics} showHeading={false}>
        <RichNarrative id="report-errors:body" className="report-analysis" label="Editar diagnóstico por classe"
          value={`## Onde os modelos ainda erram

A média global esconde assimetrias importantes. Em **CIFAR-10**, a classe mais fraca foi *cat* (F1 0,524) e a maior confusão dirigida foi *dog → cat* em 20,8% dos cães do teste. Em **FER2013**, *fear* teve F1 0,336 e a confusão dominante foi *sad → neutral* (21,6%). Em **CIFAR-100 coarse**, *small_mammals* ficou em F1 0,266.

Nos conjuntos quase saturados, os erros residuais foram raros: a maior confusão dirigida correspondeu a 0,53% no MNIST, 0,76% no KMNIST e 0,66% no GTSRB. Isso ajuda a priorizar melhorias: novos ajustes de cabeça provavelmente têm retorno limitado nessas três bases, enquanto FER2013 e as duas CIFAR ainda oferecem espaço substancial.`} />
      </ReportSection>
      <DataComponent id="report-class-diagnostics" title="Classe mais difícil e confusão dominante"
        queryId="class_diagnostics" kind="table" displayRows={diagnostics} sourceRows={diagnostics}
        description="A seta parte da classe verdadeira e aponta para a predição mais frequente fora da diagonal.">
        <DataTable rows={diagnostics} caption="Diagnóstico de erros por classe" columns={[
          { field: "dataset", label: "Dataset", presentation: "identity" },
          { field: "worstClass", label: "Classe com menor F1" },
          { field: "worstClassF1", label: "F1 da classe", renderCell: score },
          { field: "topConfusion", label: "Maior confusão" },
          { field: "topConfusionRate", label: "Taxa", renderCell: percent },
          { field: "topConfusionCount", label: "Casos", renderCell: integer },
        ]} />
      </DataComponent>
    </section>}

    {visible("report-training-behavior") && <section className="report-section">
      <ReportSection id="report-training-behavior" title="Comportamento e custo do treinamento"
        queryId="dense20_results" queryIds={["dense20_results", "experiment_summary"]}
        sourceRowsByQuery={evidence} sourceRows={results} showHeading={false}>
        <RichNarrative id="report-training-behavior:body" className="report-analysis" label="Editar análise do treino"
          value={`## Comportamento e custo do treinamento

As 900 épocas somaram **${summary.totalEpochHours?.toFixed(2)} horas** de tempo medido dentro das épocas. GTSRB consumiu **${percent(summary.gtsrbTimeShare, 1)}** desse total: imagens 128×128 reduziram o throughput para 738 exemplos/s, contra aproximadamente 9,2–10,3 mil exemplos/s nas demais bases.

Seis datasets selecionaram checkpoints entre as épocas 83 e 99, justificando o teto de 100 épocas para esta rodada. CIFAR-10, Fashion-MNIST e SVHN atingiram o melhor Macro-F1 de validação nas épocas 15, 25 e 25; completar as 100 épocas preservou o protocolo uniforme, mas aumentou custo sem melhorar a seleção final nessas três bases.

Não houve early stopping nem scheduler. A escolha sempre foi feita por **\`val_macro_f1\`**, e o conjunto de teste permaneceu isolado até o recarregamento do melhor checkpoint.`} />
      </ReportSection>
      <EvidenceChart id="report-training-cost" queryId="dense20_results" title="Custo das 100 épocas por dataset"
        description="Tempo somado das épocas; não inclui preflight, avaliação final nem exportação."
        spec={costChart} rows={costRows} sourceRows={results} height={320} />
    </section>}

    {visible("report-methods") && <ReportSection id="report-methods" title="Como a rodada foi executada"
      queryId="dense20_results" sourceRows={results} showHeading={false}>
      <RichNarrative id="report-methods:body" className="report-caveat" label="Editar metodologia"
        value={`## Como a rodada foi executada

1. **Checkpoint:** para cada dataset, o runner aceitou somente o único \`best.keras\` FP32 concluído, rejeitando QAT, dataset incorreto, formato incompatível ou fingerprint divergente.
2. **Dados:** o conjunto unido foi reconstruído e dividido de forma estratificada em 70/15/15 com seed 42. Somente o treino recebeu os exemplos extras: originais + \`floor(0,5 × N)\` amostras aumentadas.
3. **Augmentation:** brilho 0,03; contraste 0,92–1,08; translação 0,05; zoom 0,96–1,04; ruído 0,005; cutout com probabilidade 0,25 e área máxima 0,08. Flip horizontal permaneceu desligado.
4. **Arquitetura:** \`imagem → backbone até block5_pool (congelado, training=False) → GlobalAveragePooling2D → Dense(20) → LayerNormalization(ε=1e-3) → ReLU → Dense(C, logits FP32)\`. A contagem treinável foi \`2620 + 21C\`, de 2.767 a 3.607 parâmetros.
5. **Otimização:** Adam 3e-4, batch 128, FP32, sparse categorical cross-entropy com logits, 100 épocas, sem early stopping e sem scheduler.
6. **Seleção e teste:** o melhor checkpoint foi definido por Macro-F1 de validação. Depois, e somente depois, ocorreu uma única avaliação final do teste.
7. **Auditoria:** hashes antes/depois provaram que nenhum peso do backbone mudou; os nove fingerprints coincidiram; históricos, predições, matrizes de confusão, métricas, telemetria e checkpoints foram persistidos.
8. **Exportação reutilizável corrigida:** em uma passagem de inferência separada, cada checkpoint FP32 congelado foi truncado em \`block5_pool\`, seguido somente de \`GlobalAveragePooling2D\`. Isso gerou vetores N×128 antes de qualquer camada densa, sem augmentation e sem shuffle.

Nenhum módulo, dependência ou execução quântica participou desta etapa.`} />
    </ReportSection>}

    {visible("report-artifacts") && <ReportSection id="report-artifacts" title="Saídas geradas e como utilizá-las"
      queryId="features128_exports" queryIds={["features128_exports", "experiment_summary", "dense20_results"]}
      sourceRowsByQuery={evidence} sourceRows={features128Rows} showHeading={false}>
      <RichNarrative id="report-artifacts:body" className="report-analysis" label="Editar usos dos artefatos"
        value={`## Saídas geradas e como utilizá-las

Cada run preserva os artefatos do experimento Dense20 e uma exportação independente para reutilização do backbone:

- **Reprodução e auditoria:** \`manifest.json\`, \`status.json\`, histórico CSV, logs, resumo do treinamento e telemetria. Servem para reconstruir configuração, verificar runtime e explicar a escolha do checkpoint.
- **Inferência:** \`model.keras\` contém backbone congelado e classificador completo; \`best.keras\` é a seleção por validação e \`last.keras\` registra o estado final. Use \`model.keras\` quando a entrada ainda é uma imagem.
- **Entrada reutilizável para novas pontas densas:** \`encoder128.keras\` termina exatamente em \`block5_pool → GlobalAveragePooling2D\`. Ele não contém Dense, LayerNorm ou ReLU. Os arquivos \`features128/{train,val,test}_{features,labels}.npy\` somam **${integer(features128.totalRows)} vetores N×128**, com features ${features128.featureDtype}, labels ${features128.labelDtype} e ordem determinística. Eles ocupam cerca de **${features128.featureStorageMiB?.toFixed(1)} MiB** mais **${features128.labelStorageMiB?.toFixed(1)} MiB** de labels.
- **Saída específica do experimento Dense20:** \`encoder20.keras\` e \`features20/*\` permanecem válidos como embeddings produzidos por Dense(20) + LayerNorm + ReLU. Eles só servem para treinar algo depois desse gargalo; não devem ser usados para comparar pontas densas completas diferentes.

Para trocar toda a ponta densa, carregue \`train_features.npy\` e use \`Input(shape=(128,))\` como entrada do novo modelo. Ajuste arquitetura e hiperparâmetros apenas com treino/validação; carregue \`test_features.npy\` uma única vez na avaliação final. A extração N×128 registrou **${integer(features128.telemetrySamples)} amostras de telemetria** e verificou valores finitos, ordem dos labels, fingerprints e backbone inalterado nos nove datasets.

**Regra de isolamento:** ajuste hiperparâmetros apenas com treino/validação. O arquivo de teste deve continuar reservado para a comparação final.`} />
    </ReportSection>}

    {visible("report-conclusions") && <ReportSection id="report-conclusions" title="Impactos, decisão e próximos passos"
      queryId="experiment_summary" queryIds={["experiment_summary", "dense20_results", "class_diagnostics", "features128_exports"]}
      sourceRowsByQuery={{ ...evidence, class_diagnostics: diagnostics }} sourceRows={summaryRows}
      showHeading={false} className="report-conclusions">
      <RichNarrative id="report-conclusions:body" className="report-caveat" label="Editar conclusão"
        value={`## Impactos, decisão e próximos passos

**Decisão técnica corrigida:** Dense20 permanece como um benchmark clássico concluído, mas não como a entrada oficial para comparar novas pontas. A interface reutilizável correta é **N×128 após \`block5_pool → GlobalAveragePooling2D\`**, antes de qualquer Dense. Toda nova cabeça deve começar em \`Input(shape=(128,))\`.

**Onde confiar mais:** MNIST, KMNIST e GTSRB combinam Macro-F1 muito alto com erros residuais baixos. **Onde tratar como ponto de partida:** FER2013 e CIFAR-100 coarse permanecem abaixo de 0,50 Macro-F1; a compactação não resolve a limitação da representação congelada nessas bases.

**Validação recomendada antes de uma afirmação científica forte:** repetir com pelo menos 3–5 seeds, reportar média e dispersão, e comparar três controles: cabeça original, cabeça Dense20 sem augmentation extra e cabeça Dense20 com augmentation 0,5. Isso separa o efeito do gargalo do efeito do regime de dados.

**Limitações:** a média entre datasets dá o mesmo peso a bases de tamanhos e dificuldades diferentes; Macro-AUC alto não implica calibração; o teste foi usado corretamente uma vez por run, mas os deltas observados continuam sendo estimativas pontuais. Portanto, ganhos abaixo de aproximadamente 0,1 p.p. devem ser descritos como equivalência prática nesta rodada, não como vitória comprovada.`} />
    </ReportSection>}
  </article>;
}
