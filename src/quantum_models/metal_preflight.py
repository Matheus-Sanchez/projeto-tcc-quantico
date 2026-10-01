"""Strict TensorFlow/Metal readiness check used before frozen-CNN extraction."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from typing import Any, Sequence


def verify_tensorflow_gpu(tensorflow: Any | None = None) -> dict[str, Any]:
    """Require an executable TensorFlow GPU, not merely a listed device.

    Apple Metal availability depends on the current macOS graphical session. A
    process can list a PluggableDevice and still fall back to CPU when the first
    GPU kernel is dispatched. This strict matmul catches both states before an
    expensive feature extraction or QCNN run.
    """

    if tensorflow is None:
        from data_prep.data import require_tensorflow

        tensorflow = require_tensorflow()

    devices = list(tensorflow.config.list_physical_devices("GPU"))
    device_names = [str(device.name) for device in devices]
    if not devices:
        raise RuntimeError(
            "Nenhuma GPU TensorFlow foi detectada. Desbloqueie a sessão gráfica do macOS e "
            "inicie o processo no domínio Aqua antes de tentar novamente."
        )

    previous_soft_placement = tensorflow.config.get_soft_device_placement()
    try:
        tensorflow.config.set_soft_device_placement(False)
        left = tensorflow.ones((16, 16), dtype=tensorflow.float32)
        device_context = tensorflow.device("/GPU:0") if hasattr(tensorflow, "device") else nullcontext()
        with device_context:
            result = tensorflow.matmul(left, left)
    except Exception as exc:
        raise RuntimeError(
            "A GPU TensorFlow foi listada, mas falhou no teste estrito de matmul. "
            "Não inicie a extração da CNN até estabilizar a sessão gráfica/Metal."
        ) from exc
    finally:
        tensorflow.config.set_soft_device_placement(previous_soft_placement)

    result_device = str(getattr(result, "device", ""))
    if "GPU" not in result_device.upper():
        raise RuntimeError(
            "O matmul de pré-verificação não foi executado em GPU; TensorFlow faria fallback para CPU."
        )
    return {
        "tensorflow_version": str(getattr(tensorflow, "__version__", "unknown")),
        "physical_gpus": device_names,
        "matmul_device": result_device,
        "matmul_shape": [16, 16],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check as ``python -m quantum_models.metal_preflight``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        print(json.dumps(verify_tensorflow_gpu(), ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(f"Pré-verificação Metal falhou: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
