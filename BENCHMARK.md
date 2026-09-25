# Benchmark do TCC

O projeto tem quatro blocos pequenos e independentes:

- `data_prep`: leitura local, split estratificado, normalização e balanceamento;
- `classic_models`: CNN de referência em Keras;
- `quantum_models`: QCNN em PennyLane/Qiskit e integração com a CNN congelada;
- `metrics` e `utils`: relatórios, configuração e CLI.

## Ambiente

```powershell
python -m pip install -r requirements.txt
```

Os dados locais são definidos em `configs/datasets.yaml`. Nenhum comando baixa
datasets.

## Comando deste repositorio

`tcc-benchmark run --all --dry-run` pertence ao projeto anterior e nao executa
o experimento hibrido deste repositorio. Para rodar a arquitetura CNN congelada
-> QCNN -> Dense, use `python -m quantum_models.smoke` a partir da raiz deste
projeto.

## Smoke: CNN treinada e congelada → QCNN → Dense

O comando não treina uma nova CNN. Ele procura em `models/` um checkpoint
concluído equivalente ao `--dataset`, valida que ele usa `all_raw` e
`unit_interval`, expõe o vetor de 256 posições da camada
`global_pool_concat`, congela o extrator convolucional e treina somente o
circuito e a camada `Linear(256, 10)`.

```powershell
python -m quantum_models.smoke --dataset kmnist --backend pennylane --samples-per-class 12 --head-epochs 1
```

O smoke é uma validação de integração, não uma medição de acurácia final: usa
um subconjunto minúsculo e só uma época. Os resultados e checkpoints ficam em
`outputs/quantum-smoke/`.

### Teste de 50 epocas da cabeca quantica

```powershell
python -m quantum_models.smoke `
  --dataset kmnist `
  --backend pennylane `
  --model-root models `
  --samples-per-class 12 `
  --head-epochs 50 `
  --head-batch-size 2 `
  --head-train-per-class 1 `
  --head-validation-per-class 1 `
  --output-root outputs\quantum-model-backed-50ep
```

Esse comando usa o checkpoint já treinado de KMNIST, congela a sua parte
convolucional e treina apenas a QCNN e a camada densa por 50 épocas.

Por padrão, o seletor prefere `float32` e depois `relu`. No conjunto atual de
modelos, KMNIST não possui os dois atributos no mesmo checkpoint: o escolhido
é o FP32 com ativação Swish. A ativação pertence à cabeça clássica removida,
enquanto o vetor usado pelo circuito é anterior a ela. O resultado registra a
origem e o perfil real do checkpoint. Use `--strict-model-profile` somente
quando existir um checkpoint com o perfil exato desejado.

Para executar a mesma cabeça no Qiskit:

```powershell
python -m quantum_models.smoke --dataset kmnist --backend qiskit --samples-per-class 12 --head-epochs 1
```

O backend Qiskit usa simulação exata e gradiente reverso; por isso tende a ser
mais lento. Os dois backends compartilham topologia, inicialização e saída
multiclasse, e são testados quanto à equivalência numérica.

Os blocos QCNN adaptados de `takh04/QCNN` estão documentados em
`third_party/takh04-qcnn-NOTICE.md` e mantêm a licença Apache-2.0 correspondente.
