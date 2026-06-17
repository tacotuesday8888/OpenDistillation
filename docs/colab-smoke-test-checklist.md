# Colab GPU Smoke-Test Checklist

Use this checklist before marking the optional real teacher, training, or before/after comparison paths as verified. It is a manual test because this local workspace cannot provide a Colab GPU runtime.

## Setup

- [ ] Open `notebooks/opendistillation_v0_demo.ipynb` from GitHub in Colab.
- [ ] Choose **Runtime > Change runtime type > GPU** before running the training path.
- [ ] Run the setup cell and confirm it prints the expected project root.
  - Expected output in Colab: `Using project root: /content/OpenDistillation`
  - Expected status output: `OD_STATUS stage=setup status=ready`
  - Expected status log path: `/tmp/opendistillation_status.jsonl`
  - If the output pane fails, open Colab Terminal and run: `cat /tmp/opendistillation_status.jsonl`
  - If setup fails, paste the full error in `docs/colab-smoke-test-results.md`.
- [ ] Set `INSTALL_TRAINING_DEPS = True` in the optional dependency install cell.
- [ ] Run the install cell and record whether the optional Hugging Face teacher/training packages install successfully.
  - Expected skipped/default marker: `OD_STATUS stage=install status=skipped`
  - Expected opt-in markers: `OD_STATUS stage=install status=started` and `OD_STATUS stage=install status=succeeded`
  - Runtime packages checked before training: `torch, transformers, datasets, trl, peft, accelerate`
  - Packages installed by the notebook cell: `transformers<5, datasets, trl<1, peft<0.19, accelerate`
  - Expected install command: `python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate`
  - The notebook intentionally does not upgrade Colab's preinstalled GPU `torch` package.
  - If installation fails, paste the final 30-50 lines of pip output in `docs/colab-smoke-test-results.md`.
- [ ] If Colab asks for a runtime restart, restart and rerun setup before continuing.

## Anti-Invention Fact Smoke Preflight

Run this CPU-only preflight before any future anti-invention-style T4 smoke. It writes a manifest to `/tmp` and exits nonzero if the local data no longer matches the exact sample-notes GPU contract. The 2026-06-16 anti-invention run already used this gate and reached trained 2/8 exact hits; a repeat run is only useful after a local change explains why the remaining 6/8 misses still invented values.

```bash
PYTHONPATH=src python3 scripts/prepare_anti_invention_smoke.py \
  --output /tmp/opendistillation_anti_invention_smoke_manifest.json
```

- [ ] Confirm the preflight prints `Anti-invention T4 smoke preflight`.
- [ ] Confirm the stable marker starts with `OD_ANTI_INVENTION_SMOKE_MANIFEST`.
- [ ] Confirm the marker says `ready: true`.
- [ ] Confirm the report shows `Facts: 8`, `Train rows: 48`, `Held-out eval rows: 8`, `Disambiguation rows: 8`, and `known-values anti-invention rows: 8`.
- [ ] Confirm train/eval leakage is `0 exact, 0 near-duplicate`.
- [ ] Confirm the pass condition is stricter than the latest evidence when testing a new signal. The historical anti-invention gate required at least `2/8`, beating the previous best `1/8`; the current future gate requires at least `3/8` exact hits and at most `5/8` invented-value misses.
- [ ] If the repo is dirty, commit the intended code change before using the manifest as GPU evidence.
- [ ] If the preflight fails, do not run T4 training. Fix the row contract first and paste the validation errors in `docs/colab-smoke-test-results.md`.

## Default Notes Flow

- [ ] Keep `USE_SAMPLE_NOTES = True` for the default smoke test, or set it to `False` only if uploading one `.txt` or `.md` notes file.
- [ ] Confirm the notebook prints file name, extension, character count, word count, and preview.
  - Expected sample file output: `File: sample-notes.md`, `Extension: .md`, and a notes preview.
- [ ] Confirm the chunking cell prints multiple stable chunk IDs.
  - Expected sample output includes `Chunks: 4` and chunk IDs such as `chunk-0001`.
- [ ] Confirm `MockTeacherEngine` generates rows and says it does not send text to a remote endpoint.
  - Expected output includes `Teacher engine: mock-local-teacher` and `Sends text to remote endpoint: False`.
- [ ] Confirm the JSONL preview shows `instruction`, `response`, and `source_chunk_id`.
  - Expected output includes a runtime temp JSONL path, not a committed repository path.
- [ ] Confirm the dataset quality report prints separately from model quality.
  - Expected output includes row count, schema-valid rows, chunk coverage, duplicate/near-duplicate question counts, answer checks, and `OD_STATUS stage=dataset_quality status=reported`.

## Optional Real Teacher

- [ ] Keep the same Colab GPU runtime after optional Hugging Face dependencies are installed.
- [ ] Set `RUN_REAL_TEACHER = True`.
- [ ] Run the teacher cell.
  - Expected default marker: `OD_STATUS stage=teacher status=mock_started`
  - Expected opt-in marker: `OD_STATUS stage=teacher status=real_started`
  - Expected success marker: `OD_STATUS stage=teacher status=succeeded`
- [ ] Record whether `Qwen/Qwen2.5-1.5B-Instruct` starts downloading and loads successfully.
- [ ] Confirm the teacher output says `Teacher engine: huggingface-local-teacher`.
- [ ] Confirm the output says `Sends text to remote endpoint: False`.
- [ ] Confirm generated rows validate with `instruction`, `response`, and `source_chunk_id`.
- [ ] Paste the first 1-3 generated rows in `docs/colab-smoke-test-results.md`.
- [ ] If the run fails, record which plain-language failure message appears:
  - Missing optional Hugging Face package.
  - Model download or load failure.
  - CUDA/GPU memory failure.
  - Invalid JSONL/schema output.
- [ ] After the real teacher check, set `RUN_REAL_TEACHER = False` again unless intentionally training from real teacher rows.

## Optional Training

- [ ] Set `RUN_TRAINING = True`.
- [ ] Run the training cell.
  - Expected markers include `OD_STATUS stage=training status=runtime_check_finished`.
  - If training starts, expected markers include `OD_STATUS stage=training status=started` and `OD_STATUS stage=training status=succeeded`.
- [ ] Record the runtime check output, including the GPU name.
  - Expected success output includes `GPU detected: <GPU name>` and `Runtime is ready for the optional short training run.`
  - If this says no CUDA GPU is detected, confirm the Colab runtime type is GPU and paste the runtime-check output in `docs/colab-smoke-test-results.md`.
  - If this reports `operator torchvision::nms does not exist`, restart the Colab runtime and rerun the notebook without upgrading `torch`.
- [ ] Record whether `Qwen/Qwen2.5-0.5B-Instruct` starts downloading.
- [ ] Record whether the short TRL/PEFT training run starts.
- [ ] Record total runtime.
- [ ] Confirm the adapter output path prints under `outputs/notes-lora/adapter`.
  - Expected success output includes `Adapter output: /content/OpenDistillation/outputs/notes-lora/adapter`.
- [ ] If the run fails, record the exact error and whether the notebook's failure message explains the next step.
  - For memory failures, paste the exact CUDA out-of-memory line and the notebook's recovery message.

## Before/After Model Quality Report

- [ ] Run the comparison cell after training succeeds.
  - Expected markers include `OD_STATUS stage=comparison status=configured`, `OD_STATUS stage=comparison status=started`, and `OD_STATUS stage=comparison status=succeeded`.
- [ ] Confirm comparison questions and generated reference answers are shown. Current fact-ledger runs can show up to 8 held-out fact questions; fallback generated-question runs may show fewer.
- [ ] Confirm both base-model answers and trained-adapter answers are shown with crude reference-overlap values.
  - Expected headings include `Model quality report`, `Question:`, `Reference answer:`, `Base answer`, and `Trained adapter answer`.
- [ ] Confirm the notebook labels this as a qualitative smoke report, not a benchmark.
- [ ] Record whether comparison quality is visibly different, unchanged, or broken.
  - Paste the question and both answers in `docs/colab-smoke-test-results.md`.

## Artifact Safety

- [ ] Confirm generated datasets, adapters, model files, checkpoints, and caches are under ignored runtime/output paths.
- [ ] Confirm no notebook outputs are saved before committing.
- [ ] Confirm no API keys, tokens, `.env` files, or local machine config are created or committed.

## Results Paste Template

Copy this block into `docs/colab-smoke-test-results.md` after the run:

```text
Colab runtime type:
GPU type:
Anti-invention manifest marker:
Anti-invention manifest path:
Dependency install result:
Real teacher run: skipped/succeeded/failed
Real teacher model download/load result:
Real teacher generated rows:
Dataset quality report:
Status log markers:
Model download result:
Training starts: yes/no
Adapter output created: yes/no
Adapter output path:
Before/after model quality report output: yes/no
Runtime:
Peak memory or memory failure:
Exact error messages:
Questions used for comparison:
Base model answers:
Trained adapter answers:
Reference-overlap values:
Exact fact-hit report:
Fact miss diagnostic report:
Quality verdict versus latest best trained exact hits 2/8 and invented-value misses 6/8:
```
