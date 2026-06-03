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

## Before/After Comparison

- [ ] Run the comparison cell after training succeeds.
  - Expected markers include `OD_STATUS stage=comparison status=configured`, `OD_STATUS stage=comparison status=started`, and `OD_STATUS stage=comparison status=succeeded`.
- [ ] Confirm the question and generated reference answer are shown.
- [ ] Confirm both base-model answer and trained-adapter answer are shown.
  - Expected headings: `Question:`, `Reference answer from generated dataset:`, `Base model answer:`, and `Trained adapter answer:`.
- [ ] Confirm the notebook labels this as a qualitative sanity check, not a benchmark.
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
Dependency install result:
Real teacher run: skipped/succeeded/failed
Real teacher model download/load result:
Real teacher generated rows:
Status log markers:
Model download result:
Training starts: yes/no
Adapter output created: yes/no
Adapter output path:
Before/after comparison output: yes/no
Runtime:
Peak memory or memory failure:
Exact error messages:
Question used for comparison:
Base model answer:
Trained adapter answer:
```
