# OpenDistillation

> A personal model factory for the AI PC and AI phone era.

OpenDistillation is an open-source project for helping people build small personal models they can understand, improve, and run locally.

The long-term idea is bigger than one document tool. As AI PCs and AI phones become normal, people will want more than one general assistant. They will want small personal models for specific parts of life:

- **Notes / school model** - learns from class notes, study material, and personal knowledge.
- **Coding model** - learns a developer's project patterns, snippets, and preferred style.
- **Writing model** - learns tone, outlines, drafts, and editing preferences.
- **Work model** - learns team docs, repeated workflows, and role-specific tasks.
- **Phone model** - learns lightweight personal routines that can run close to the device.

OpenDistillation should become the open-source workflow for making those models. The first version is intentionally much smaller:

> Upload TXT/MD notes in Colab, generate inspectable training examples with a safe mock teacher by default or an opt-in local Qwen teacher, and optionally start a tiny LoRA fine-tuning run from that dataset.

## Why This Exists

Fine-tuning and distillation tools are powerful, but they still feel like training infrastructure. Most people do not want a research framework first. They want a practical way to make a small model for a specific part of their life.

OpenDistillation focuses on the product workflow:

1. Pick a personal model type.
2. Bring the right source material.
3. Turn that material into training examples.
4. Train or adapt a small student model.
5. Compare the result.
6. Export or run it locally.

The first model type is the notes / school model because it is easy to understand, safe to scope, and works with plain text.

## Current Status

OpenDistillation is in foundation and prototype mode.

What exists now:

- Public positioning and project docs.
- A scoped v0 notes-model Colab flow.
- A runnable prototype notebook for TXT/MD loading, chunking, dataset validation, dataset quality reporting, mock QA generation, and opt-in real teacher generation.
- Minimal Python helpers under `src/opendistillation/`.
- Helper interfaces for teacher, training, comparison, and future export engines.
- An optional local open-source teacher engine using `Qwen/Qwen2.5-1.5B-Instruct` through Hugging Face Transformers.
- A bounded optional training engine using `Qwen/Qwen2.5-0.5B-Instruct`, TRL `SFTTrainer`, and PEFT LoRA.
- A before/after comparison helper that can compare held-out sample-fact questions or chunk-diverse generated questions with trained-adapter answers after opt-in training.
- A fact-rich sample-notes experiment with 24 mock rows and four held-out sample-fact questions for checking note-grounded answer movement.
- A deterministic fact-ledger builder that extracts explicit `Label: value` facts plus safe bullet/list facts, creates separate train/eval rows, checks exact and near-duplicate train/eval leakage, and scores exact expected-term hits.
- A six-row-per-fact label/value training signal with same-chunk disambiguation and anti-invention known-values rows, so each sample fact gets explicit `Label: value` supervision, contrast against nearby labels, and a recorded-values-only prompt before optional training.
- Runtime checks and plain-language setup messages for optional Hugging Face, CUDA, teacher, training, and comparison failures.
- Notebook `OD_STATUS` markers and a runtime status log so Colab output-frame failures do not erase the state of long optional cells.
- Internal fact metadata now follows held-out comparison rows through source-chunk reordering, so exact expected-term scoring is tied to the fact/question identity instead of list position.
- A local SFT preview shows the exact prompt/completion text that optional TRL training will receive before any GPU work starts.
- A first-demo implementation plan.
- A manual Colab GPU smoke-test checklist.
- A smoke-test results file that records the first real Colab T4 blockers, a clean GitHub-opened T4 training/comparison pass, one real-teacher end-to-end T4 verification, the uploaded-notes rehearsal status, two earlier multi-question Colab GPU quality smoke results, the 30-step sample-fact CLI T4 smoke, and the latest fact-ledger CLI T4 smoke.
- GitHub issue forms and a starter issue plan.
- Guardrails to avoid committing generated datasets, checkpoints, model weights, or secrets.

What does not exist yet:

- Evidence that the tiny adapter meaningfully improves answers on the fact-rich sample. The latest GPU evidence is the 2026-06-16 same-chunk disambiguation fact-ledger Colab T4 smoke: 48 fact-ledger train rows, 16 disambiguation rows, 8 held-out eval questions, and exact fact hits stayed at base 0/8 and trained 0/8. This failed the quality rule because the trained adapter changed all 8 answers but missed every checked fact.
- Multiple personal model profiles.
- Coding, writing, work, or phone model flows.
- A CLI.
- A model export implementation.
- A SaaS, account system, Mac app, or cloud backend.

## V0 Scope

The v0 prototype is only the first personal model type:

> A notes / school model from one `.txt` or `.md` file.

The first prototype should prove one complete path:

> A user uploads one `.txt` or `.md` notes file in Colab, sees generated question-answer training data, can opt into a short small-student fine-tune, and later compares the base and trained model and gets explicit local-run instructions.

Current prototype constraints:

- **Input:** `.txt` and `.md` notes only.
- **Teacher:** deterministic local mock teacher by default; optional local `Qwen/Qwen2.5-1.5B-Instruct` teacher after `RUN_REAL_TEACHER = True`.
- **Dataset:** JSONL rows with `instruction`, `response`, and `source_chunk_id`.
- **Training:** optional short TRL/PEFT LoRA entry point; skipped by default and verified once in a clean GitHub-opened Colab T4 runtime.
- **Quality:** deterministic dataset checks for row count, chunk coverage, duplicate questions, answer length, missing fields, and source chunk IDs, plus a fact-ledger quality gate for safe fact extraction, train/eval leakage, and exact expected-term coverage.
- **Comparison:** optional bounded base-vs-adapter quality report after training; skipped by default and verified locally with fake model dependencies. For fact-ledger notes, the report now uses direct held-out fact-ledger eval questions about concrete facts such as `Glass Harbor`, `copper-lantern-47`, `llama-harbor-alpha`, `4:17 PM`, and `ultramarine`, with expected terms carried by internal fact metadata. For uploaded notes without a passing fact ledger, it falls back to generated questions from distinct source chunks first. The report uses PEFT's adapter-disabled inference path for the base answer. Exact miss diagnostics now explain whether wrong trained answers look like same-chunk value confusion, other known-value confusion, invented numeric/time/identifier values, label echo, or answer-shape learning without the fact. The latest disambiguation Colab T4 smoke changed 8/8 answers but scored 0/8 exact facts, so changed wording is still not useful note learning.
- **Export:** placeholder only; no GGUF or local runtime output yet.

Remaining v0 constraints after the current prototype:

- **Student:** one recommended small model around 0.5B-1.5B parameters.
- **Training:** response distillation / supervised fine-tuning.
- **Export:** GGUF if practical; otherwise document the exact next command and limitation.
- **Local runtime:** llama.cpp and/or Ollama-style instructions.

Out of scope for v0:

- Multiple model profiles.
- Coding, writing, work, or phone model implementations.
- PDF parsing.
- Arbitrary document ingestion.
- Dashboards.
- Accounts or billing.
- Cloud training orchestration.
- Novel distillation algorithm claims.
- Large-scale benchmark suites.

## First Colab Flow

The planned notes-model notebook flow is specified in [`docs/first-demo-flow.md`](docs/first-demo-flow.md). In short:

1. Open the notebook from GitHub.
2. Use the sample notes file or upload one `.txt` / `.md` notes file.
3. Preview and validate the text.
4. Split it into short chunks.
5. Generate question-answer examples with the mock teacher, or opt into the local Qwen teacher.
6. Preview, quality-check, and save the JSONL dataset in the notebook runtime.
7. Build a fact ledger, train rows, held-out eval questions, and a leakage report.
8. Show an optional short training plan that stays skipped by default.
9. If training runs, compare held-out sample-fact questions for the committed sample notes, or chunk-diverse generated questions for uploaded notes.
10. Show clear placeholders for export and local running.

The current notebook is [`notebooks/opendistillation_v0_demo.ipynb`](notebooks/opendistillation_v0_demo.ipynb). Its default path runs without GPU, package installs, model downloads, paid APIs, remote APIs, or training. When opened from GitHub in Colab, the setup cell clones this repository before importing local helpers and creates `/tmp/opendistillation_status.jsonl` for recoverable status markers. The default sample-notes path now generates 24 fact-aware mock rows from four short chunks, prints held-out sample-fact comparison questions, and shows a fact-ledger quality gate with train/eval leakage checks. When that fact-ledger gate passes, optional training now uses the six-row-per-fact label/value fact-ledger train rows, including same-chunk disambiguation and known-values-only anti-invention rows that list real note values and warn against invented substitutes. It also prints a bounded SFT preview, and optional comparison uses held-out fact-ledger eval rows with internal fact metadata for exact-hit scoring plus fact-miss diagnostics. The optional real teacher and optional training cells require a Colab GPU runtime and the Hugging Face packages installed by the notebook. The clean GitHub-opened T4 smoke tests for mock-teacher training/comparison and real-teacher end-to-end wiring are recorded in [`docs/colab-smoke-test-results.md`](docs/colab-smoke-test-results.md). That file also records the first uploaded-notes rehearsal, where both `.txt` and `.md` upload paths passed through validation, chunking, mock teacher, dataset save, training skipped, and comparison skipped. The latest GPU evidence is the 2026-06-16 same-chunk disambiguation fact-ledger T4 smoke: 8 facts, 48 train rows, 16 disambiguation rows, 8 held-out eval rows, a 30-step LoRA adapter, and exact fact hits stayed at 0/8. Useful note learning is still not proven.

## Repository Map

```text
OpenDistillation/
  README.md                         # public project page
  START_HERE.md                     # quick orientation for future work
  AGENTS.md                         # working agreements for coding agents
  .github/
    ISSUE_TEMPLATE/                 # structured GitHub issue forms
  docs/
    colab-smoke-test-checklist.md   # manual GPU verification checklist
    colab-smoke-test-results.md     # recorded GPU smoke-test status
    current-decisions.md            # decisions that should not be reopened casually
    dataset-schema.md               # JSONL shape for generated examples
    engine-integration-points.md    # where real engines plug in later
    first-demo-flow.md              # exact v0 Colab user flow
    first-demo-implementation-plan.md
    github-issue-plan.md
    github-launch-checklist.md
    product-vision.md
    roadmap.md
  examples/
    sample-notes.md                 # tiny input file for the first demo
  notebooks/
    README.md
    opendistillation_v0_demo.ipynb  # runnable v0 prototype notebook
  src/
    opendistillation/               # prototype helper package
```

## Development Direction

The current prototype covers the safe first slice of the notes / school model:

1. Load and validate `.txt` or `.md`.
2. Preview the notes.
3. Chunk text with stable IDs.
4. Generate QA examples with the deterministic mock teacher by default.
5. Validate and serialize the JSONL dataset.
6. Optionally generate real QA rows with the local Qwen teacher.
7. Extract a fact ledger and create separate train/eval rows with leakage checks.
8. Prepare a short optional LoRA fine-tuning request from the validated dataset rows.
9. Prepare an optional bounded before/after quality report from held-out sample-fact questions or chunk-diverse generated questions.

The next work should use the local fact-miss diagnostics to make a targeted row-signal change after the failed disambiguation T4 smoke. Do not broaden beyond the notes model, chase training knobs, or build export before the demo flow can show useful note-grounded answers.

Future personal model types should reuse the same broad workflow, but they should not be implemented until the notes model path works.

## Contributing

The project is not ready for broad contributions yet. The best early contributions are small and concrete:

- Clarify the README or beginner docs.
- Test the default and opt-in Colab notes-model flows.
- Improve the real teacher prompt and validation without broadening v0.
- Help keep v0 small enough to ship.

Use the issue forms in `.github/ISSUE_TEMPLATE/` when this repository is published.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
