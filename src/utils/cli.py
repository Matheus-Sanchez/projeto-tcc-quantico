"""Comandos leves para conferir os dados locais antes dos experimentos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from data_prep.audit import audit_local_dataset, write_audit_report
from .experiment import DATASET_ORDER, DEFAULT_DATASET_REGISTRY, DEFAULT_OUTPUT_ROOT, load_dataset_registry


def _add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_DATASET_REGISTRY,
        help="Registro YAML de caminhos locais (padrão: configs/datasets.yaml).",
    )


def _add_dataset_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASET_ORDER, help="Dataset a processar.")
    group.add_argument("--all", action="store_true", help="Processa todos os datasets do registro, sequencialmente.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcc-benchmark", description="Auditoria e preparação de dados locais para o benchmark do TCC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets", help="Lista os datasets e caminhos configurados.")
    _add_registry_argument(datasets)

    audit = subparsers.add_parser("audit", help="Audita datasets locais; não baixa dados nem carrega TensorFlow.")
    _add_registry_argument(audit)
    _add_dataset_selector(audit)
    audit.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    audit.add_argument("--max-samples", type=int, default=None, help="Limita a auditoria para diagnóstico rápido.")
    audit.add_argument("--labels-only", action="store_true", help="Confere apenas rótulos e distribuição.")
    audit.add_argument("--no-hash", action="store_true", help="Não calcula hashes de imagens duplicadas.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_dataset_registry(args.registry)
        if args.command == "datasets":
            for name, entry in registry.items():
                print(f"{name}: {entry.root} ({entry.adapter})")
            return 0

        selected = registry.keys() if args.all else (args.dataset,)
        exit_code = 0
        for name in selected:
            entry = registry[name]
            verify_images = not args.labels_only
            report = audit_local_dataset(
                entry.adapter, entry.root, max_samples=args.max_samples,
                verify_images=verify_images, hash_images=verify_images and not args.no_hash, **entry.options,
            )
            destination = args.output_root / name / "audit" / "audit.json"
            write_audit_report(report, destination)
            print(f"{name}: {report.checked_samples} exemplos, {report.error_count} erro(s), {report.warning_count} aviso(s)")
            print(f"  relatório: {destination}")
            exit_code = max(exit_code, 0 if report.is_healthy else 1)
        return exit_code
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
