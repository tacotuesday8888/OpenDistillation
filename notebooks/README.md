# Notebooks

This directory holds the Colab-first OpenDistillation demo.

Current notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The current notebook proves the first safe part of the notes / school model flow:

> Load a `.txt` or `.md` notes file, preview and chunk it, generate deterministic mock training examples, show a dataset quality report, prepare an optional short TRL/PEFT LoRA training run, and prepare a bounded before/after quality report that stays skipped by default.

It also includes an opt-in local real teacher path with `Qwen/Qwen2.5-1.5B-Instruct`. Keep `RUN_REAL_TEACHER = False` unless running a Colab GPU smoke test with the optional Hugging Face packages installed.

## Notebook Rules

- One notes-model path, not a menu of personal model profiles.
- Small defaults.
- Clear output after every major step.
- Honest labels for anything not implemented yet.
- No committed generated datasets, checkpoints, model weights, or secrets.

## Current Notebook Target

With training skipped by default, the notebook should run top to bottom on CPU and show:

- Introduction and status warning.
- Upload/loading and validation for `.txt` and `.md` notes.
- Text preview.
- Chunk preview.
- Mock teacher generation by default with factual recall, explanation, flashcard, and misconception-check question styles.
- Deterministic dataset quality report for row count, chunk coverage, duplicate or near-duplicate questions, answer length, missing fields, and source chunk IDs.
- Optional real teacher generation with `RUN_REAL_TEACHER = False` by default.
- Optional dependency install section with `INSTALL_TRAINING_DEPS = False` by default.
- Optional student fine-tuning section with `RUN_TRAINING = False` by default.
- Runtime readiness messages before any opt-in training starts.
- `OD_STATUS` markers and a runtime status log at `/tmp/opendistillation_status.jsonl` in Colab.
- Optional multi-question before/after quality report that skips when training is skipped.
- Placeholder section for export.

Future engines should plug in through the interfaces described in `docs/engine-integration-points.md`.

The optional real teacher, training, and comparison sections are not part of the CPU smoke path. They require a Colab GPU runtime and installing the Hugging Face training packages listed in the notebook. Training saves adapters under `outputs/`, which is ignored by git. Use `docs/colab-smoke-test-checklist.md` before marking that GPU path verified. If the Colab output pane fails, open the Colab Terminal and run `cat /tmp/opendistillation_status.jsonl`.

As of 2026-06-03, the optional sample-notes training/comparison path has passed once from a clean GitHub-opened Colab T4 runtime, and the optional real-teacher path has passed one end-to-end T4 wiring check from sample notes through comparison. The first uploaded-notes rehearsal now has one `.txt` pass and one `.md` pass through validation, chunking, mock teacher rows, dataset save, training skipped, and comparison skipped. The new deterministic dataset quality report is verified locally in the default notebook path; the new multi-question model-quality report still needs a fresh Colab GPU smoke run after opt-in training. See `docs/colab-smoke-test-results.md` for exact package versions, adapter paths, memory notes, output-rendering caveats, historical upload-control blockers, and before/after output.
