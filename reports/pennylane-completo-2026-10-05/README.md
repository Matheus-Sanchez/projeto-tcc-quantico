# Relatório completo PennyLane — 05/10/2026

Leia [RELATORIO_COMPLETO.md](RELATORIO_COMPLETO.md). O pacote resume os nove treinamentos concluídos e inclui as tabelas de evidência em `evidence/` e as figuras principais usadas no relatório.

A galeria completa reúne 632 figuras em PNG e SVG, o catálogo e a página HTML. Baixe `figuras-completas.zip`, extraia-o nesta pasta e abra `figures/index.html`. Os 74 gráficos embutidos no Markdown também estão disponíveis diretamente em `figures/`.

## Evidências e limites

- `evidence/` contém históricos, resultados por classe, matrizes de confusão, condições de ruído/shots, estatísticas de hardware, amostras e resumos de batches/tensores e verificações.
- `evidence/metadata.json` não foi incluído porque contém caminhos absolutos desta máquina.
- Os arquivos originais das execuções estão em `outputs/quantum-parallel-128-20/runs/`; os checkpoints, predições completas e telemetria bruta não fazem parte deste pacote.
- A prévia interativa local e o snapshot de aproximadamente 304 MiB não foram incluídos. O relatório Markdown e a galeria ZIP são a entrega portátil desta branch.

## Reproduzir tabelas e gráficos

Os scripts de análise não treinam modelos. Em ambiente de análise separado, com as execuções originais disponíveis localmente, instale `requirements-report.txt` e execute na raiz deste pacote:

```sh
python build_evidence.py
python render_figures.py
```

A galeria ZIP representa o conjunto completo de figuras da apuração de 05/10/2026; para atualizá-la depois de regenerar os gráficos, compacte novamente a pasta `figures/` como `figuras-completas.zip`.

A campanha foi concluída nos nove datasets, com 100 épocas por execução. Não houve inferência em hardware quântico, e a comparação clássica permaneceu inelegível por falta de `split_fingerprint` nos configs clássicos.
