#!/usr/bin/env python3
"""Train one VulGuard Lite model from local files or a Hugging Face dataset."""

import argparse
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
from vulguard_lite.models.init_model import models
from vulguard_lite.training import training
from vulguard_lite.utils.reproducibility import seed_everything


DEEP_MODELS = {"deepjit", "jitfine", "simcom"}
DICTIONARY_MODELS = {"deepjit", "simcom"}
PAIRED_MODELS = {"jitfine", "simcom"}


def existing_components(value):
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def validate_args(args, parser):
    if not args.train_set and not args.hf_repo_id:
        parser.error("provide --train-set or --hf-repo-id")
    if args.model in {"jitfine", "simcom"} and not args.val_set and not args.hf_repo_id:
        parser.error(f"--val-set is required for local {args.model} training")
    if args.model in DICTIONARY_MODELS and not args.dictionary and not args.hf_repo_id:
        parser.error(f"--dictionary is required for local {args.model} training")

    for option_name in ("train_set", "val_set"):
        value = getattr(args, option_name)
        if not value:
            continue
        paths = existing_components(value)
        expected = 2 if args.model in PAIRED_MODELS else 1
        if len(paths) != expected:
            parser.error(
                f"--{option_name.replace('_', '-')} expects {expected} file(s) for {args.model}; "
                "separate paired files with a comma"
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            parser.error(f"input file not found: {', '.join(missing)}")

    for option_name in ("dictionary", "hyperparameters"):
        value = getattr(args, option_name)
        if value and not Path(value).is_file():
            parser.error(f"--{option_name.replace('_', '-')} not found: {value}")
    if args.model_path and not Path(args.model_path).exists():
        parser.error(f"--model-path not found: {args.model_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a VulGuard Lite JIT vulnerability-prediction model."
    )
    parser.add_argument("--model", required=True, choices=models)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--repo-language", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--sampling-seed", type=int, default=None)

    parser.add_argument("--train-set", help="JSONL path; paired models use file1,file2")
    parser.add_argument("--val-set", help="JSONL path; paired models use file1,file2")
    parser.add_argument("--dictionary", help="DeepJIT/SimCom dictionary JSONL")
    parser.add_argument("--hyperparameters", help="Override bundled hyperparameters JSON")

    parser.add_argument("--output-dir", dest="dg_save_folder", default="./output")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", dest="resume_from_checkpoint", action="store_true")

    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-split-path", default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.seed < 0 or (args.sampling_seed is not None and args.sampling_seed < 0):
        parser.error("seeds must be non-negative")

    if args.model in DEEP_MODELS and args.hyperparameters is None:
        args.hyperparameters = str(
            PACKAGE_ROOT / "vulguard_lite" / "models" / args.model / "hyperparameters.json"
        )

    # Attributes retained by training.py from the original common CLI.
    args.mode = "local"
    args.repo_path = None
    args.repo_clone_url = None
    args.repo_clone_path = None
    args.model_output_dir = None
    args.sampling_run_id = None

    validate_args(args, parser)
    seed_everything(args.seed)
    result = training(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


