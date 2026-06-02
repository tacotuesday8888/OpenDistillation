# Colab GPU Smoke-Test Checklist

Use this checklist before marking the optional training and before/after comparison path as verified. It is a manual test because this local workspace cannot provide a Colab GPU runtime.

## Setup

- [ ] Open `notebooks/opendistillation_v0_demo.ipynb` from GitHub in Colab.
- [ ] Choose **Runtime > Change runtime type > GPU** before running the training path.
- [ ] Run the setup cell and confirm it prints the expected project root.
- [ ] Set `INSTALL_TRAINING_DEPS = True` in the optional dependency install cell.
- [ ] Run the install cell and record whether `torch`, `transformers`, `datasets`, `trl`, `peft`, and `accelerate` install successfully.
- [ ] If Colab asks for a runtime restart, restart and rerun setup before continuing.

## Default Notes Flow

- [ ] Use `examples/sample-notes.md` or upload one `.txt` notes file.
- [ ] Confirm the notebook prints file name, extension, character count, word count, and preview.
- [ ] Confirm the chunking cell prints multiple stable chunk IDs.
- [ ] Confirm `MockTeacherEngine` generates rows and says it does not send text to a remote endpoint.
- [ ] Confirm the JSONL preview shows `instruction`, `response`, and `source_chunk_id`.

## Optional Training

- [ ] Set `RUN_TRAINING = True`.
- [ ] Run the training cell.
- [ ] Record the runtime check output, including the GPU name.
- [ ] Record whether `Qwen/Qwen2.5-0.5B-Instruct` starts downloading.
- [ ] Record whether the short TRL/PEFT training run starts.
- [ ] Record total runtime.
- [ ] Confirm the adapter output path prints under `outputs/notes-lora/adapter`.
- [ ] If the run fails, record the exact error and whether the notebook's failure message explains the next step.

## Before/After Comparison

- [ ] Run the comparison cell after training succeeds.
- [ ] Confirm the question and generated reference answer are shown.
- [ ] Confirm both base-model answer and trained-adapter answer are shown.
- [ ] Confirm the notebook labels this as a qualitative sanity check, not a benchmark.
- [ ] Record whether comparison quality is visibly different, unchanged, or broken.

## Artifact Safety

- [ ] Confirm generated datasets, adapters, model files, checkpoints, and caches are under ignored runtime/output paths.
- [ ] Confirm no notebook outputs are saved before committing.
- [ ] Confirm no API keys, tokens, `.env` files, or local machine config are created or committed.
