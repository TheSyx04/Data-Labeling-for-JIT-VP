# Script for training models for Just-in-time Vulnerability Prediction

## Requirements

- Python 3.9 or later
- CPU for `lapredict`, `lr`, and `tlel`
- An NVIDIA GPU is recommended for `deepjit`, `jitfine`, and `simcom`
- Internet access or a local Hugging Face cache is required when JITFine first
  loads `microsoft/codebert-base`

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA training, install the PyTorch build matching the machine's CUDA
version before running `pip install -e .`.

Check the training command:

```bash
python scripts/train.py --help
```

After installation, the same command is available as:

```bash
vulguard-train --help
```

## Training with Local Data

### LAPredict

```bash
python scripts/train.py \
  --model lapredict \
  --repo-name openssl \
  --repo-language C \
  --train-set /data/train_Kamei_features_openssl.jsonl \
  --output-dir ./output \
  --seed 42
```

### Logistic Regression

```bash
python scripts/train.py \
  --model lr \
  --repo-name openssl \
  --repo-language C \
  --train-set /data/train_Kamei_features_openssl.jsonl \
  --output-dir ./output \
  --seed 42
```

### TLEL

```bash
python scripts/train.py \
  --model tlel \
  --repo-name openssl \
  --repo-language C \
  --train-set /data/train_Kamei_features_openssl.jsonl \
  --sampling \
  --sampling-seed 42 \
  --output-dir ./output
```

### DeepJIT

```bash
python scripts/train.py \
  --model deepjit \
  --repo-name openssl \
  --repo-language C \
  --device cuda \
  --train-set /data/train_merge_openssl.jsonl \
  --dictionary /data/dict_openssl.jsonl \
  --epochs 30 \
  --output-dir ./output \
  --seed 42
```

### JITFine

JITFine expects comma-separated feature and code files. Both files must contain
the same commits.

```bash
python scripts/train.py \
  --model jitfine \
  --repo-name openssl \
  --repo-language C \
  --device cuda \
  --train-set /data/train_Kamei_features_openssl.jsonl,/data/train_merge_openssl.jsonl \
  --val-set /data/val_Kamei_features_openssl.jsonl,/data/val_merge_openssl.jsonl \
  --epochs 50 \
  --output-dir ./output \
  --seed 42
```

### SimCom

SimCom expects comma-separated Kamei feature and patch files.

```bash
python scripts/train.py \
  --model simcom \
  --repo-name openssl \
  --repo-language C \
  --device cuda \
  --train-set /data/train_Kamei_features_openssl.jsonl,/data/train_patch_openssl.jsonl \
  --val-set /data/val_Kamei_features_openssl.jsonl,/data/val_patch_openssl.jsonl \
  --dictionary /data/dict_openssl.jsonl \
  --epochs 30 \
  --output-dir ./output \
  --seed 42
```

## Training with a Hugging Face Dataset

The command can resolve training files, validation files, and dictionaries from
a public Hugging Face dataset repository:

```bash
python scripts/train.py \
  --model jitfine \
  --repo-name openssl \
  --repo-language C \
  --device cuda \
  --hf-repo-id TheSyx/vulguard_lite \
  --hf-revision main \
  --hf-split-path dataset/openssl/openssl_3_1 \
  --epochs 50 \
  --output-dir ./output \
  --seed 42
```

Use a dataset commit hash instead of `main` for a fixed dataset revision. For a
private dataset, download the files first and use `--train-set`, `--val-set`,
and `--dictionary`.

## Dataset Format

Input files use JSON Lines format: one JSON object per line.

### Kamei Features

The following format is used by LAPredict, logistic regression, TLEL, JITFine,
and the Sim component of SimCom:

```json
{"commit_id":"abc123","label":1,"ns":2,"nd":1,"nf":3,"entropy":0.5,"la":42,"ld":7,"lt":1000,"fix":1,"ndev":2,"age":10,"nuc":1,"exp":20,"rexp":4,"sexp":3}
```

LAPredict reads only `la`. Logistic regression and TLEL read all 14 features.

### Code Changes

DeepJIT, JITFine, and the Com component of SimCom use records containing commit
messages and code changes:

```json
{"commit_id":"abc123","label":1,"messages":"fix bounds check","code_change":"<ADD> if (n < size) <REMOVE> old_call()"}
```

JITFine requires both `<ADD>` and `<REMOVE>` markers in `code_change`.

### DeepJIT and SimCom Dictionary

The dictionary is a two-line JSONL file:

1. The first line maps message tokens to integer IDs.
2. The second line maps code tokens to integer IDs.

Both mappings must include the `<NULL>` token.

## Undersampling

Add `--sampling` to balance the two label classes in the training set:

```bash
python scripts/train.py \
  --model tlel \
  --repo-name openssl \
  --repo-language C \
  --train-set /data/train_Kamei_features_openssl.jsonl \
  --sampling \
  --sampling-seed 42
```

For paired JITFine and SimCom inputs, commit IDs are sampled once and the same
selection is applied to both files.

## Hyperparameters

Default hyperparameters are stored in:

- `vulguard_lite/models/deepjit/hyperparameters.json`
- `vulguard_lite/models/jitfine/hyperparameters.json`
- `vulguard_lite/models/simcom/hyperparameters.json`

Use `--hyperparameters /path/custom.json` to override the default file.

## Output

Model files and checkpoints are written to:

```text
<output-dir>/dg_cache/save/<repo-name>/models/<model>_seed_<seed>/
├── best_epoch/
├── checkpoints/
└── last_epoch/
```

Use `--output-dir` to change the output root.

## Resuming Training

Run the same command with `--resume`:

```bash
python scripts/train.py \
  --model deepjit \
  --repo-name openssl \
  --repo-language C \
  --device cuda \
  --train-set /data/train_merge_openssl.jsonl \
  --dictionary /data/dict_openssl.jsonl \
  --epochs 30 \
  --output-dir ./output \
  --seed 42 \
  --resume
```

Use `--checkpoint-dir /path/to/checkpoints` to specify a different checkpoint
directory.

## Recommended Checks

Before starting a long training run:

1. Run `python scripts/train.py --help`.
2. Check that every JSONL record contains `commit_id` and `label`.
3. Check that paired files contain the same commit IDs.
4. Run one epoch to verify the input paths and schema.
5. For GPU training, verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
