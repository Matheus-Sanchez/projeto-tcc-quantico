import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const pct = v => v == null ? "—" : `${(v * 100).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
const num = v => Number(v).toLocaleString("pt-BR");
const hours = v => v == null ? "—" : `${Math.floor(v)}h ${Math.round((v % 1) * 60)}min`;
const moment = v => v ? new Date(v).toLocaleString("pt-BR", { timeZone: "America/Sao_Paulo", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—";
const col = (field, label, renderCell) => ({ field, label, ...(renderCell ? { renderCell } : {}) });

function EvidenceTable({ id, queryId, title, rows, columns, sourceRows = rows }) {
  return <DataComponent id={id} queryId={queryId} title={title} kind="table" displayRows={rows} sourceRows={sourceRows}>
    <DataTable rows={rows} columns={columns} searchable={false} compactNumbers={false} caption={title} rowKey="dataset" />
  </DataComponent>;
}

export function ReportContent() {
  const { snapshot, appTitle, setAppTitle, canEdit, mode, visible } = useDataApp();
  const rows = q => snapshot.queries[q].rows;
  const [o] = rows("overview");
  const runs = rows("runs"), completed = runs.filter(r => r.status === "completed"), started = runs.filter(r => r.confirmedEpochs > 0);
  const active = runs.find(r => r.status === "running"), pending = runs.filter(r => r.status === "pending");
  const labels = Object.fromEntries(runs.map(r => [r.dataset, r.label]));
  const em = runs.find(r => r.dataset === "emnist_balanced");
  const fashion = runs.find(r => r.dataset === "fashion_mnist");
  const cifarCompleted = completed.filter(r => ["cifar10", "cifar100_coarse"].includes(r.dataset));
  const [dataset, setDataset] = useState(active?.dataset || completed.at(-1)?.dataset || runs[0].dataset);
  const [noiseDataset, setNoiseDataset] = useState("fashion_mnist");
  const histories = rows("history").filter(r => r.dataset === dataset);
  const curveRows = histories.map(r => ({ epoch: r.epoch, treinoRate: r.accuracy, validacaoRate: r.val_accuracy, macroF1Rate: r.val_macro_f1 }));
  const noiseSource = rows("noise_aggregate").filter(r => r.dataset === noiseDataset);
  const noiseChart = noiseSource.map(r => ({ noiseScale: r.noiseScale, shotsLabel: r.shotsLabel, accuracyRate: r.accuracy }));
  const emTele = rows("telemetry").filter(r => r.dataset === em.dataset);
  const currentTele = rows("telemetry").filter(r => r.dataset === active?.dataset);
  const noiseStress = rows("noise_aggregate").filter(r => r.noiseScale === 2 && r.shots === 256);
  const observed = moment(o.observedAt);
  const allComplete = completed.length === runs.length;
  const latestCompletion = completed.map(r => r.completedAt).filter(Boolean).sort().at(-1);
  const finalCompleted = completed.filter(r => ["svhn", "gtsrb", "fer2013"].includes(r.dataset));
  const finalInventory = rows("inventory").filter(r => finalCompleted.some(run => run.dataset === r.dataset));
  const fer = runs.find(r => r.dataset === "fer2013");
  const summaryLead = allComplete ? `A execução terminou em **${moment(latestCompletion)}**, com 100 épocas, teste completo e estudo de ruído para cada dataset.` : `${active ? `O ${active.label} segue ativo. ` : ""}${pending.length ? `${pending.map(r => r.label).join(", ")} aguardam a execução sequencial.` : ""}`;
  const summary = `## Resumo do treinamento\n\n**${o.completedRuns} dos ${o.totalRuns} treinamentos foram concluídos**. ${summaryLead}\n\n- Acurácia final de teste: ${completed.map(r => `**${r.label} ${pct(r.testAccuracy)}**`).join("; ")}.\n${active ? `- ${active.label}: **${active.confirmedEpochs}/${active.plannedEpochs} épocas confirmadas**; época ${active.currentEpoch} em andamento. O melhor macro-F1 de validação até a leitura é **${pct(active.bestValMacroF1)}**, na época ${active.bestEpoch}.\n` : ""}- Foram coletadas **${o.noiseConditions} condições de ruído e shots**, com testes em ${num(o.testExamplesCompleted)} exemplos nos ${o.completedRuns} datasets concluídos.`;
  const progressText = allComplete ? `Leitura em **${observed} (horário de São Paulo)**. Foram confirmadas **${o.confirmedEpochs} de ${o.plannedEpochs} épocas (100%)**. Os nove datasets têm teste e estudo de ruído concluídos. A última execução terminou em **${moment(latestCompletion)}**; nenhum processo de treinamento estava ativo na leitura.` : `Leitura em **${observed} (horário de São Paulo)**. O processo Python ${o.activeProcess ? "estava ativo" : "não foi encontrado"}. Foram confirmadas ${o.confirmedEpochs} de ${o.plannedEpochs} épocas planejadas (${pct(o.confirmedEpochs / o.plannedEpochs)} por contagem de épocas). Cada dataset tem um custo diferente; esse percentual não estima o tempo restante. Este relatório registra uma leitura pontual.`;

  return <article className="report-content quantum-report" aria-label="Relatório do treinamento quântico">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? e => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
    </header>
    {visible("report-summary") && <ReportSection id="report-summary" queryId="overview" queryIds={["overview", "runs"]}
      sourceRowsByQuery={{ overview: rows("overview"), runs: started }} title="Resumo do treinamento" showHeading={false}>
      <RichNarrative id="report-summary:body" value={summary} />
    </ReportSection>}

    <section className="report-section">
      {visible("training-progress") && <DataComponent id="training-progress" queryId="runs" title="Etapa de cada treinamento" kind="custom" displayRows={runs} sourceRows={runs}>
        <div className="quantum-progress" data-reviewed-rows>
          {runs.map(r => <div key={r.dataset} className={`quantum-progress-row is-${r.status}`}>
            <div className="quantum-progress-label"><strong>{r.label}</strong><span>{r.status === "completed" ? "Concluído" : r.status === "running" ? "Em execução" : "Na fila"}</span></div>
            <div className="quantum-progress-track"><progress max={r.plannedEpochs} value={r.confirmedEpochs} aria-label={`${r.label}: ${r.confirmedEpochs} de ${r.plannedEpochs} épocas confirmadas`} /><strong>{r.confirmedEpochs}/{r.plannedEpochs}</strong></div>
            <div className="quantum-progress-note">{r.status === "completed" ? `Teste e ruído concluídos · ${hours(r.durationHours)} · encerrado ${moment(r.completedAt)}` : r.status === "running" ? `Época ${r.currentEpoch} · ${r.currentSavedBatches}/${r.batchesPerEpoch} lotes salvos na leitura · teste ainda não coletado` : "Features disponíveis; ainda sem treino, checkpoints ou teste quântico."}</div>
          </div>)}
        </div>
      </DataComponent>}
      {visible("progress-note") && <ReportSection id="progress-note" queryId="overview" queryIds={["overview", "runs"]} title="Cobertura da leitura" sourceRowsByQuery={{ overview: rows("overview"), runs: completed }} showHeading={false}>
        <RichNarrative id="training-progress:note" value={progressText} />
      </ReportSection>}
      {cifarCompleted.length > 0 && visible("cifar-results") && <ReportSection id="cifar-results" title="CIFAR: resultados concluídos" queryId="runs" sourceRows={cifarCompleted} showHeading={false}>
        <RichNarrative id="cifar-results:body" value={`## CIFAR: novos resultados de teste\n\n${cifarCompleted.map(r => `**${r.label}** terminou em **${moment(r.completedAt)}**, após **${hours(r.durationHours)}**. O checkpoint da época **${r.bestEpoch}**, escolhido pelo macro-F1 de validação de **${pct(r.bestValMacroF1)}**, alcançou **${pct(r.testAccuracy)} de acurácia** e **${pct(r.testMacroF1)} de macro-F1** nos ${num(r.test)} exemplos de teste. O treino completou 100 épocas e o estudo de ruído concluiu 64 condições.`).join("\n\n")}\n\nNa última época, a diferença entre acurácia de treino e validação foi de ${cifarCompleted.map(r => `**${((r.lastTrainAccuracy-r.lastValAccuracy)*100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })} pontos percentuais (${r.label})**`).join(" e ")}. O teste utiliza os checkpoints selecionados pela validação, anteriores à época 100.`} />
      </ReportSection>}
    </section>

    <section className="report-section">
      {visible("result-table") && <EvidenceTable id="result-table" queryId="runs" title="Resultados finais sobre o conjunto completo de teste" rows={completed}
        columns={[col("label", "Dataset"), col("confirmedEpochs", "Épocas"), col("bestEpoch", "Melhor época"), col("bestValMacroF1", "F1 de validação", pct), col("testAccuracy", "Acurácia de teste", pct), col("testMacroF1", "Macro-F1 de teste", pct)]} />}
      {visible("generalization") && <ReportSection id="generalization" title="O teste usa o checkpoint escolhido na validação" queryId="runs" queryIds={["runs", "protocol"]} sourceRowsByQuery={{ runs: completed, protocol: rows("protocol") }} showHeading={false}>
        <RichNarrative id="generalization:body" value={`Os ${completed.length} modelos concluídos completaram as 100 épocas. Para avaliar o teste, o código recarrega o checkpoint com maior macro-F1 de validação: ${completed.map(r => `**época ${r.bestEpoch} (${r.label})**`).join("; ")}.\n\nNo Fashion-MNIST, a última época alcançou **${pct(fashion.lastTrainAccuracy)} de acurácia de treino** e **${pct(fashion.lastValAccuracy)} na validação**, diferença de ${((fashion.lastTrainAccuracy - fashion.lastValAccuracy) * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })} pontos percentuais. A seleção pela validação preserva o modelo da época ${fashion.bestEpoch}. As configurações mantêm 100 épocas, sem parada antecipada nem redução automática da taxa de aprendizado.`} />
      </ReportSection>}
    </section>

    <section className="report-section">
      {visible("emnist-detail") && <ReportSection id="emnist-detail" queryId="runs" queryIds={["runs", "telemetry"]}
        sourceRowsByQuery={{ runs: [em], telemetry: emTele }} title="EMNIST Balanced: resultado concluído" showHeading={false}>
        <RichNarrative id="emnist-detail:body" value={`## EMNIST Balanced: treinamento, teste e ruído concluídos\n\nA execução terminou em **${moment(em.completedAt)}**, após **${hours(em.durationHours)}**, com 100 épocas, ${num(em.train)} exemplos de treino e ${em.classes} classes. O checkpoint da **época ${em.bestEpoch}** foi selecionado pelo macro-F1 de validação de **${pct(em.bestValMacroF1)}**.\n\nNo conjunto completo de ${num(em.test)} exemplos de teste, a acurácia foi **${pct(em.testAccuracy)}** e o macro-F1 foi **${pct(em.testMacroF1)}**. Foram salvas as predições, métricas por classe, matriz de confusão e 64 condições do estudo de ruído.`} />
      </ReportSection>}
      {active && visible("active-detail") && <ReportSection id="active-detail" queryId="runs" queryIds={["runs", "telemetry"]}
        sourceRowsByQuery={{ runs: [active], telemetry: currentTele }} title={`${active.label}: progresso e última validação`} showHeading={false}>
        <RichNarrative id="active-detail:body" value={`## ${active.label}: ${active.confirmedEpochs} épocas salvas, época ${active.currentEpoch} em andamento\n\nO treinamento começou em **${moment(active.startedAt)}**, com ${num(active.train)} exemplos de treino e ${active.classes} classes. Na leitura, havia **${active.currentSavedBatches} de ${active.batchesPerEpoch} lotes de treino salvos na época ${active.currentEpoch}**; ela ainda não tinha checkpoint confirmado. A retomada usa a **última época completa (${active.confirmedEpochs})**.\n\nNa época ${active.confirmedEpochs}: acurácia de treino **${pct(active.lastTrainAccuracy)}**, loss **${active.lastTrainLoss.toFixed(4)}**; acurácia de validação **${pct(active.lastValAccuracy)}**, macro-F1 **${pct(active.lastValMacroF1)}**, loss **${active.lastValLoss.toFixed(4)}**. O melhor macro-F1 até a leitura foi **${pct(active.bestValMacroF1)} na época ${active.bestEpoch}**. A diferença entre a acurácia de treino e validação na última época é de **${((active.lastTrainAccuracy - active.lastValAccuracy) * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })} pontos percentuais**; o aumento da acurácia de treino não indica melhora equivalente na validação.\n\nTempo da sessão até a leitura: **${hours(active.durationHours)}**. As últimas cinco épocas levaram mediana de **${(active.recentEpochSeconds/60).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} minutos para treino e validação**; a consolidação dos diagnósticos e do checkpoint ocorre depois. Ainda faltam ${active.plannedEpochs - active.confirmedEpochs} épocas confirmadas, além do teste e estudo de ruído. Não há resultado de teste final para esse dataset.`} />
      </ReportSection>}
      {finalCompleted.length > 0 && visible("final-completions") && <ReportSection id="final-completions" queryId="runs" queryIds={["runs", "inventory"]}
        sourceRowsByQuery={{ runs: finalCompleted, inventory: finalInventory }} title="SVHN, GTSRB e FER2013: resultados concluídos" showHeading={false}>
        <RichNarrative id="final-completions:body" value={`## SVHN, GTSRB e FER2013: resultados finais\n\n${finalCompleted.map(r => `**${r.label}** terminou em **${moment(r.completedAt)}**, após **${hours(r.durationHours)}**. O checkpoint da **época ${r.bestEpoch}**, selecionado pelo macro-F1 de validação de **${pct(r.bestValMacroF1)}**, alcançou **${pct(r.testAccuracy)} de acurácia** e **${pct(r.testMacroF1)} de macro-F1** nos ${num(r.test)} exemplos de teste. Foram concluídas 100 épocas e ${finalInventory.find(i => i.dataset === r.dataset).noiseConditions} condições de ruído.`).join("\n\n")}\n\nNo FER2013, a última época alcançou **${pct(fer.lastTrainAccuracy)} no treino** e **${pct(fer.lastValAccuracy)} na validação**, diferença de **${((fer.lastTrainAccuracy-fer.lastValAccuracy)*100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })} pontos percentuais**. A acurácia mais alta do treino não se mantém na validação. O resultado final de teste usa o checkpoint selecionado pela validação, da época ${fer.bestEpoch}.`} />
      </ReportSection>}
      {visible("training-curves") && <EvidenceChart id="training-curves" queryId="history" title="Acurácia e macro-F1 por época confirmada"
        rows={curveRows} sourceRows={histories} height={330}
        spec={{ type: "line", x: "epoch", y: "treinoRate", fields: ["treinoRate", "validacaoRate", "macroF1Rate"], stackable: false, startAtZero: false, xLabel: "Época", valueDecimals: 2, legend: { labels: { treinoRate: "Acurácia de treino", validacaoRate: "Acurácia de validação", macroF1Rate: "Macro-F1 de validação" } } }}
        headerControls={<Dropdown label="Dataset" showLabel value={dataset} choices={started.map(r => r.dataset)} choiceLabels={labels} onChange={setDataset} />} />}
    </section>

    <section className="report-section">
      {visible("collected-data") && <ReportSection id="collected-data" queryId="overview" queryIds={["overview", "inventory"]}
        sourceRowsByQuery={{ overview: rows("overview"), inventory: rows("inventory") }} title="Dados já coletados" showHeading={false}>
        <RichNarrative id="collected-data:body" value={`## Dados já coletados\n\n- **${num(o.confirmedEpochs)} registros de épocas confirmadas**: loss, acurácia, macro-F1 e métricas por classe na validação, taxa de aprendizado e duração.\n- **${num(o.trainBatchJson)} JSONs de lotes de treino** e **${num(o.validationBatchJson)} de validação**. Cada arquivo contém quatro registros de circuito: estatísticas de entradas e saídas, gradientes, atualização de pesos, tempos e avaliações lógicas. ${active ? `A contagem inclui lotes parciais do ${active.label}.` : "Todos os lotes pertencem a épocas completas."}\n- **${num(o.telemetrySamples)} amostras de telemetria**: CPU, memória e disco do processo e dos filhos, com coleta configurada a cada cinco segundos.\n- **Checkpoints e sondas quânticas** nos ${started.length} datasets iniciados. Os ${completed.length} concluídos também têm predições de teste (rótulos, logits e probabilidades), métricas por classe e matriz de confusão.\n- **${o.noiseConditions} condições de ruído e shots** concluídas: 64 por dataset, com predições, contagens X/Y/Z, erros de leitura, fidelidade, distância de traço, pureza e entropia.`} />
      </ReportSection>}
      {visible("inventory-table") && <EvidenceTable id="inventory-table" queryId="inventory" title="Inventário por treinamento iniciado" rows={rows("inventory")}
        columns={[col("label", "Dataset"), col("trainBatchJson", "Lotes de treino"), col("validationBatchJson", "Lotes de validação"), col("probeJson", "Arquivos de sondas"), col("noiseConditions", "Condições de ruído"), col("testPredictions", "Predições de teste", v => v ? "Salvas" : "Ainda não")]} />}
      {visible("telemetry-table") && <EvidenceTable id="telemetry-table" queryId="telemetry" title="Tempo e telemetria por sessão" rows={rows("telemetry")}
        columns={[col("label", "Dataset"), col("elapsedHours", "Tempo da sessão", hours), col("samples", "Amostras"), col("peakMemoryMbTree", "Pico RAM dos processos", v => `${(v/1024).toFixed(2)} GiB`), col("lastAt", "Última amostra", moment)]} />}
    </section>

    <section className="report-section">
      {visible("noise-summary") && <ReportSection id="noise-summary" queryId="noise_aggregate" title="Estudo de ruído" sourceRows={rows("noise_aggregate")} showHeading={false}>
        <RichNarrative id="noise-study:intro" value={`## Estudo de ruído nos ${completed.length} modelos concluídos\n\nForam usadas escalas de ruído 0, 0,5, 1 e 2; leitura exata e 256, 1.024 ou 4.096 shots. Cada condição com shots tem cinco seeds (42 a 46). O estudo reutiliza o melhor checkpoint e aplica simulação local sobre todo o teste. Na escala 2 com 256 shots, as quedas médias de acurácia foram (valor negativo indica aumento): ${noiseStress.map(r => `**${r.label}: ${r.degradationPp.toLocaleString("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 })} ponto percentual**`).join("; ")}.`} />
      </ReportSection>}
      {visible("noise-chart") && <EvidenceChart id="noise-chart" queryId="noise_aggregate" title="Acurácia de teste por escala de ruído e shots" rows={noiseChart} sourceRows={noiseSource} height={300}
        spec={{ type: "line", x: "noiseScale", y: "accuracyRate", series: "shotsLabel", stackable: false, startAtZero: false, valueDecimals: 3, xLabel: "Escala de ruído" }}
        headerControls={<Dropdown label="Dataset" showLabel value={noiseDataset} choices={completed.map(r => r.dataset)} choiceLabels={labels} onChange={setNoiseDataset} />} />}
      {visible("noise-table") && <EvidenceTable id="noise-table" queryId="noise_aggregate" title="Escala 2 e 256 shots: média das cinco seeds" rows={rows("noise_aggregate").filter(r => r.noiseScale === 2 && r.shots === 256)}
        columns={[col("label", "Dataset"), col("accuracy", "Acurácia média", pct), col("minAccuracy", "Mínimo", pct), col("maxAccuracy", "Máximo", pct), col("degradationPp", "Queda (p.p.)", v => v.toLocaleString("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 }))]} />}
    </section>

    <section className="report-section">
      {visible("protocol") && <ReportSection id="protocol" queryId="features" queryIds={["features", "protocol"]} title="Features e protocolo" sourceRowsByQuery={{ features: rows("features"), protocol: rows("protocol") }} showHeading={false}>
        <RichNarrative id="protocol:body" value={`## Features e protocolo\n\nOs nove datasets têm **features de 128 valores float32 por exemplo**, com ${num(o.totalInputExamples)} exemplos no total entre treino, validação e teste. Todos passaram pela validação inicial do runner antes do primeiro treinamento. Os ${started.length} já iniciados registram zero NaN e zero infinito em seus arquivos de estatísticas.\n\nA cabeça segue **features128 → Dense(128, ReLU) → Dense(20, linear) → quatro circuitos de cinco qubits → logits**. São 180 parâmetros quânticos (45 por circuito), com três camadas de rotações e emaranhamento. O backend é **Qiskit Aer local em CPU**, com quatro workers persistentes; treino por parameter-shift, Adam, learning rate 0,0003, batch 128, seed 42 e 100 épocas. A cabeça de dez classes possui 19.482 parâmetros treináveis no total.`} />
      </ReportSection>}
      {visible("features-table") && <EvidenceTable id="features-table" queryId="features" title="Splits lidos das features salvas" rows={rows("features")}
        columns={[col("label", "Dataset"), col("classes", "Classes"), col("train", "Treino"), col("val", "Validação"), col("test", "Teste"), col("total", "Total")]} />}
      <RichNarrative id="report:methods" className="report-disclosure" value={`**Fontes e limites.** Leitura local dos arquivos de configuração, status, checkpoints, histórico, métricas de teste, diagnósticos e telemetria em ${observed}. Os cabeçalhos dos 54 arrays NPY foram conferidos; o histórico foi reconciliado com os checkpoints e os ${completed.length} resultados com as métricas de teste. As fontes e as definições podem ser consultadas nos menus de cada componente.\n\nA referência clássica gravada nos ${started.length} runs não é elegível para comparação (faltam status concluído e métricas de teste nessa referência). Os resultados deste relatório descrevem a cabeça com Qiskit; não estabelecem ganho sobre a cabeça clássica. Um único seed de treinamento foi executado por dataset. ${allComplete ? "Os nove treinamentos foram concluídos. " : o.activeProcess ? "O treinamento permanece ativo. " : ""}O relatório registra uma leitura pontual.`} />
    </section>
  </article>;
}
