# Evidências compactas — treinamento Mac (29–30/09/2026)

Este diretório contém as curvas completas de 100 épocas e os resultados finais dos nove treinamentos Dense independentes executados no Mac com seed 42, além dos registros compactos de ambiente e telemetria resumida. Os horários e a interpretação estão no relatório raiz `RELATORIO_DADOS_TREINAMENTO_2026-09-29_A_2026-09-30.md`.

Cada pasta de dataset inclui `history.csv`, `test_metrics.json`, `classification_report.json`, `feature_statistics.json`, `timing.json`, `status.json` e `telemetry/summary.json`. `preflight.json` registra macOS, Python, TensorFlow, dtype e dispositivos vistos no preflight.

Os vetores de features e seus manifests/hash são versionados em `outputs/classic-dense20/` via Git LFS. Os pesos dos modelos, previsões, logits, matrizes redundantes, amostras individuais de telemetria e diretórios das execuções QCNN incompletas permanecem nos outputs locais do Mac e não foram incluídos nesta publicação.
