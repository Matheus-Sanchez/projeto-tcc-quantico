# Artefatos de treinamento PennyLane — 05/10/2026

Pacote dos nove jobs concluídos (seed 42, 100 épocas cada). `model-results/` contém os arquivos de configuração e resultado por dataset, históricos completos, métricas de teste, predições, todos os checkpoints salvos e os arquivos de matriz. Os caminhos absolutos da máquina foram substituídos por `<local-path-redacted>` nas cópias textuais; a coleta original em `outputs/` não foi alterada.

`archives/` contém os diretórios completos `diagnostics/` e `telemetry/` de cada execução, incluindo arquivos de batch, sondas, estudo de ruído/counts e amostras brutas de hardware. São 10,38 GiB compactados. Os nove pacotes são tar.gz; o de EMNIST foi dividido em duas partes para respeitar o limite por arquivo do Git LFS.

## Baixar os arquivos LFS

Instale Git LFS e, depois de clonar o repositório, execute:

```sh
git lfs install
git lfs pull
```

Sem Git LFS, os arquivos grandes aparecem como ponteiros de texto.

## Conferir e extrair

Da raiz do repositório, confira os hashes:

```sh
cd training-artifacts/quantum-parallel-128-20
shasum -a 256 -c SHA256SUMS.txt
```

Extraia, por exemplo, o pacote de CIFAR-10:

```sh
mkdir -p recovered-runs/cifar10
tar -xzf archives/cifar10-diagnostics-telemetry.tar.gz -C recovered-runs/cifar10
```

Para EMNIST Balanced, reconstitua o tar.gz antes de extrair:

```sh
cat archives/emnist_balanced-diagnostics-telemetry.tar.gz.part-01 archives/emnist_balanced-diagnostics-telemetry.tar.gz.part-02 > emnist_balanced.tar.gz
mkdir -p recovered-runs/emnist_balanced
tar -xzf emnist_balanced.tar.gz -C recovered-runs/emnist_balanced
```

Os demais datasets seguem o mesmo padrão de CIFAR-10, trocando o nome do arquivo e da pasta.

Consulte [o relatório resumido](../../reports/RELATORIO_TREINAMENTO_QUANTICO_PENNYLANE_2026-10-05.md) e [o relatório completo com figuras](../../reports/pennylane-completo-2026-10-05/RELATORIO_COMPLETO.md).
