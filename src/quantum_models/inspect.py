"""Read saved quantum diagnostics without training or starting simulators."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from quantum_models.persistence import read_json


def inspect_run(run, probe="best"):
    run = Path(run).expanduser().resolve()
    config = read_json(run/"config.json")
    if not config:
        raise FileNotFoundError(f"No quantum run configuration in {run}")
    report = {
        "run": str(run), "dataset": config["dataset"], "backend": config["backend"],
        "mode": config["mode"], "status": read_json(run/"status.json").get("status"),
        "checkpoint": read_json(run/"checkpoints"/"state.json"),
        "profile": {key: value for key, value in read_json(run/"profile.json").items()
                    if key != "branches"},
        "result": read_json(run/"result.json"),
    }
    probe_file = run/"diagnostics"/"probes"/f"{probe}.npz"
    if probe_file.is_file():
        with np.load(probe_file, allow_pickle=False) as arrays:
            report["probe"] = {
                "name": probe, "arrays": {name: {"shape": list(arrays[name].shape),
                                              "dtype": str(arrays[name].dtype)}
                                         for name in arrays.files},
                "probability_sum_max_error": float(np.max(np.abs(arrays["probabilities_z"].sum(-1)-1))),
                "global_purity_mean": float(arrays["purity_global"].mean()),
                "global_entropy_bits_mean": float(arrays["entropy_global_bits"].mean()),
                "subsystem_one_qubit_entropy_bits_mean": float(arrays["entropy_one_qubit_bits"].mean()),
            }
    study = run/"diagnostics"/"noise-study"
    selection = read_json(study/"selection.json")
    if selection:
        report["noise_study"] = read_json(study/selection["generation"]/"summary.json")
        report["noise_conditions_csv"] = str(study/selection["generation"]/"conditions.csv")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--probe", default="best")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_run(args.run, args.probe), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
