# Resultados quânticos locais — SVHN, GTSRB e FER2013

Este diretório preserva os seis treinamentos completos com seed 42, 100 épocas
por run e os backends PennyLane e Qiskit. As execuções ocorreram entre
8 e 12 de outubro de 2026, em Windows/Python 3.12, com simuladores locais em CPU.

## Resultados no conjunto de teste

| Dataset | Backend | Época selecionada | Acurácia | Macro-F1 |
|---|---|---:|---:|---:|
| SVHN | PennyLane | 9 | 93,01% | 92,63% |
| SVHN | Qiskit | 9 | 93,01% | 92,63% |
| GTSRB | PennyLane | 84 | 99,64% | 99,57% |
| GTSRB | Qiskit | 84 | 99,64% | 99,57% |
| FER2013 | PennyLane | 96 | 50,85% | 48,50% |
| FER2013 | Qiskit | 96 | 50,80% | 48,46% |

Os valores completos estão em [summary.csv](summary.csv), derivados dos
`result.json` de cada run. A seleção usa o maior Macro-F1 na validação.
Todos os treinamentos completaram 100 épocas e 64 condições do estudo de ruído.

## Conteúdo preservado

- `archives/`: um `.tar.gz` por treinamento, armazenado via Git LFS. Cada arquivo
  contém integralmente a run original: configuração, pesos iniciais, checkpoints,
  histórico, predições, métricas, diagnósticos por batch/época, sondas, estudo de
  ruído e telemetria. O smoke inicial de MNIST/PennyLane tem arquivo separado.
- `summaries/`: cópias legíveis dos históricos, configurações, resultados,
  ambiente, métricas e estados dos checkpoints dos seis treinamentos completos.
- `inventories/`: um inventário compactado por arquivo, com caminho original,
  tamanho e SHA-256 de cada item. Todo conteúdo compactado foi comparado com os
  arquivos originais durante a preparação.
- [manifest.json](manifest.json): proveniência, tamanhos, contagens e SHA-256 dos
  arquivos compactados e inventários.
- `original-matrices/`: metadados originais das matrizes completa e smoke. Como
  cada dataset foi executado em uma chamada separada, a matriz completa original
  registra somente FER2013. `summary.csv` e `manifest.json` consolidam os seis jobs.

As métricas do smoke são apenas um teste de integração e ficam separadas das
métricas completas. Os dados de entrada/features128 continuam no layout original
do projeto; não são duplicados nestes arquivos de resultados.

## Baixar e restaurar os arquivos originais

Na raiz do repositório, com Git LFS instalado, execute no PowerShell:

```powershell
git lfs pull --include="results/quantum-parallel-128-20/archives/*.tar.gz"

$destination = ".\outputs\quantum-parallel-128-20"
New-Item -ItemType Directory -Path $destination -Force | Out-Null
Get-ChildItem ".\results\quantum-parallel-128-20\archives\*.tar.gz" | ForEach-Object {
    tar.exe -xzf $_.FullName -C $destination
    if ($LASTEXITCODE -ne 0) { throw "Falha ao extrair $($_.Name)" }
}
Copy-Item ".\results\quantum-parallel-128-20\original-matrices\runs\*" "$destination\runs\" -Force
Copy-Item ".\results\quantum-parallel-128-20\original-matrices\smoke\*" "$destination\smoke\" -Force
```

A extração restaura os caminhos `runs/<dataset>/<backend>/seed-42/` e o smoke
original. Execute em uma cópia sem treinamento ativo, pois arquivos de mesmo
nome serão substituídos. Os caminhos absolutos históricos nos metadados foram
preservados como evidência da execução.

Para verificar o SHA-256 dos arquivos baixados, compare o resultado abaixo com
as entradas `sha256` do manifesto:

```powershell
Get-FileHash ".\results\quantum-parallel-128-20\archives\*.tar.gz" -Algorithm SHA256
```

O empacotamento e a verificação completa podem ser reproduzidos a partir dos
resultados locais com `python scripts/archive-quantum-results.py` (somente
biblioteca padrão e `tar` do sistema).
