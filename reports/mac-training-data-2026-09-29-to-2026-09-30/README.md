# Evidências compactas — treinamento Mac (29–30/09/2026)

Este diretório contém as curvas completas de 100 épocas e os resultados finais dos nove treinamentos Dense independentes executados no Mac com seed 42, além dos registros compactos do ambiente e da telemetria resumida. Os vetores congelados usados como entrada foram pré-computados em WSL2/Linux; essa etapa está separada no relatório raiz `RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md`.

Cada pasta de dataset inclui `history.csv`, `test_metrics.json`, `classification_report.json`, `feature_statistics.json`, `timing.json`, `status.json` e `telemetry/summary.json` dos treinos Dense no Mac. `preflight.json` registra macOS, Python, TensorFlow, dtype e dispositivos vistos no preflight do Mac. Nos caches de features em `outputs/classic-dense20/`, os manifests e registros compactos de hardware/telemetria indicam que a extração dos vetores ocorreu antes em WSL2/Linux.

Os arrays de features e rótulos são versionados em `outputs/classic-dense20/` via Git LFS; manifests, status e registros compactos de ambiente/telemetria ficam no Git. Os pesos dos modelos, previsões, logits, matrizes redundantes e amostras individuais de telemetria permanecem nos outputs locais e não foram incluídos nesta publicação. As amostras individuais do WSL2 não são parte do treino Dense no Mac.
