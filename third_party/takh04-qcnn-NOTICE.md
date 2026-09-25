# Origem da QCNN e alterações locais

Os blocos `U_SU4`, `Pooling_ansatz1` e a conectividade em
`src/quantum_models/qcnn.py` foram adaptados de
[takh04/QCNN](https://github.com/takh04/QCNN), commit
`9091189e198cb9d50c5e938b224a17bcfb2c14f9`, sob Apache-2.0.

Referências de origem: `QCNN/unitary.py` e `QCNN/QCNN_circuit.py`.
O artigo correspondente é Tak Hur, Leeseok Kim e Daniel K. Park, “Quantum
convolutional neural network for classical data classification”, *Quantum
Machine Intelligence* 4, 3 (2022), https://doi.org/10.48550/arXiv.2108.00661.

Alterações deste projeto: leitura das probabilidades dos oito qubits, saída
multiclasse `Linear(256, C)`, validação de entrada, integração PyTorch e o
adaptador de convolução congelada. Essas extensões não são código original dos
autores nem reproduzem resultados multiclasse publicados por eles.
