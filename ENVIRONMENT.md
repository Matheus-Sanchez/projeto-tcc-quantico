# Ambiente comum para os experimentos

Este projeto usa CPython **3.12.3** em 64 bits no Windows x86_64 e no macOS
ARM64. As dependências diretas dos três braços experimentais estão fixadas
em `requirements.txt`; ambos os sistemas devem instalar exatamente esse arquivo.

## Instalação

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

No macOS, use o equivalente:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

No macOS ARM64, o marcador de plataforma em `requirements.txt` e no extra
`tensorflow` instala `tensorflow-metal==1.2.0`. Esse plugin é específico ao
backend Metal; não altera as versões dos frameworks, bibliotecas quânticas ou
do código comum. Antes de iniciar uma extração ou treino da CNN, confirme que
`tf.config.list_physical_devices("GPU")` retorna uma GPU; não execute o
benchmark em CPU como substituto caso a sessão não exponha Metal.

Para rodadas híbridas no Mac, a enumeração isolada não basta: uma sessão
gráfica instável pode listar Metal e falhar no primeiro kernel. Use a
pré-verificação estrita incluída no projeto, de preferência em um Terminal do
usuário gráfico atual (Aqua), e só então inicie o experimento:

```bash
# Execute a partir da raiz do repositório.
./scripts/run_hybrid_macos.sh --help
```

O launcher se reexecuta no domínio Aqua, só passa se um `matmul` estrito for
materializado em `GPU:0` e inclui `--require-tensorflow-gpu` automaticamente.
A QCNN PennyLane/Qiskit continua em CPU/`float64` por desenho; Metal acelera a
extração inicial da CNN congelada, não o simulador quântico atual.

## Protocolo de comparação

Use `torch==2.8.0` nos três classificadores comparados: cabeça clássica,
`qml.qnn.TorchLayer` (PennyLane) e `EstimatorQNN` com `TorchConnector`
(Qiskit Machine Learning). Assim, as cabeças recebem as mesmas features,
loss, otimizador e versões de bibliotecas; somente a cabeça varia.

`pennylane-qiskit` não foi incluído: ele serve para usar Qiskit como backend
do PennyLane, e não para a comparação independente entre os dois braços.

TensorFlow 2.18.1 permanece disponível para o pipeline clássico/histórico.
Em Windows nativo, essa versão não oferece GPU CUDA; use WSL2 para esse
backend. Isso não impede a comparação controlada das três cabeças PyTorch,
mas hardware, backend e precisão devem ser registrados nos resultados.
