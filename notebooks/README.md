# Notebooks

This directory holds the Colab-first OpenDistillation demo.

Current notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The current notebook proves the first safe part of the notes / school model flow:

> Load a `.txt` or `.md` notes file, preview and chunk it, generate deterministic mock training examples, show a dataset quality report, prepare an optional short TRL/PEFT LoRA training run, and prepare a bounded before/after quality report that stays skipped by default.

It also now shows a deterministic fact-ledger quality gate before training. That gate extracts explicit `Label: value` facts and safe bullet/list facts, builds six label/value train rows per fact plus separate held-out eval rows, checks train/eval leakage, and reports exact expected-term coverage without loading a model.

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
- Mock teacher generation by default with fact-aware study rows for simple `Label: value` notes and the generic excerpt fallback for normal uploaded notes.
- Deterministic dataset quality report for row count, chunk coverage, duplicate or near-duplicate questions, answer length, missing fields, and source chunk IDs.
- Deterministic fact-ledger quality report for extracted facts, six-row-per-fact train/eval row counts, fact coverage, train/eval leakage, and strict expected-term checks.
- Four held-out sample-fact comparison questions when the committed `examples/sample-notes.md` file is loaded.
- Optional real teacher generation with `RUN_REAL_TEACHER = False` by default.
- Optional dependency install section with `INSTALL_TRAINING_DEPS = False` by default.
- Optional student fine-tuning section with `RUN_TRAINING = False` by default.
- Bounded SFT preview that prints the exact prompt/completion rows before any optional training starts.
- Runtime readiness messages before any opt-in training starts.
- `OD_STATUS` markers and a runtime status log at `/tmp/opendistillation_status.jsonl` in Colab.
- Optional before/after quality report that skips when training is skipped. When the fact-ledger gate passes, optional training uses fact-ledger train rows and comparison uses held-out fact-ledger eval questions with internal fact metadata for exact-hit scoring. The committed sample-notes fallback uses held-out fact questions; uploaded notes without fact-ledger rows use chunk-diverse generated questions.
- Placeholder section for export.

Future engines should plug in through the interfaces described in `docs/engine-integration-points.md`.

The optional real teacher, training, and comparison sections are not part of the CPU smoke path. They require a Colab GPU runtime and installing the Hugging Face training packages listed in the notebook. Training saves adapters under `outputs/`, which is ignored by git. Use `docs/colab-smoke-test-checklist.md` before marking that GPU path verified. If the Colab output pane fails, open the Colab Terminal and run `cat /tmp/opendistillation_status.jsonl`.

As of the current fact-ledger flow, the optional sample-notes training/comparison path has passed once from a clean GitHub-opened Colab T4 runtime, and the optional real-teacher path has passed one end-to-end T4 wiring check from sample notes through comparison. The first uploaded-notes rehearsal now has one `.txt` pass and one `.md` pass through validation, chunking, mock teacher rows, dataset save, training skipped, and comparison skipped. The deterministic dataset quality report is verified locally in the default notebook path. The current sample-notes default produces 4 chunks, 24 schema-valid fact-aware rows, 4/4 chunk coverage, zero duplicate or near-duplicate questions, zero answer-length warnings, and held-out sample-fact questions while training and comparison stay skipped by default. The same safe path now reports a fact-ledger split with 8 extracted facts, 48 fact train rows, 8 held-out eval rows, 16 same-chunk disambiguation rows, zero exact train/eval leaks, zero near-duplicate leaks, and zero missing expected terms. If that gate passes and the user opts into training, the notebook trains from the six-row-per-fact fact-ledger train rows, prints an SFT preview, and later compares against held-out fact-ledger eval questions with expected terms tied to fact metadata. The latest GPU evidence is the failed 2026-06-16 disambiguation fact-ledger T4 smoke: 48 train rows, 16 disambiguation rows, 8 held-out eval rows, changed trained-adapter answers, and base/trained scores of 0/8 exact facts. Useful note learning is not proven unless exact held-out fact hits improve. See `docs/colab-smoke-test-results.md` for exact package versions, adapter paths, memory notes, historical upload-control blockers, GPU-limit evidence, and before/after evidence.
