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

> Upload TXT/MD notes in Colab, generate mock training examples, and optionally start a tiny LoRA fine-tuning run from that dataset.

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

OpenDistillation is in foundation and prototype-skeleton mode.

What exists now:

- Public positioning and project docs.
- A scoped v0 notes-model Colab flow.
- A runnable prototype skeleton notebook for TXT/MD loading, chunking, dataset validation, and mock QA generation.
- Minimal Python helpers under `src/opendistillation/`.
- Helper interfaces for future teacher and export engines.
- A bounded optional training engine using `Qwen/Qwen2.5-0.5B-Instruct`, TRL `SFTTrainer`, and PEFT LoRA.
- A before/after comparison helper that can compare one base-model answer with one trained-adapter answer after opt-in training.
- Runtime checks and plain-language setup messages for the optional Colab GPU path.
- A first-demo implementation plan.
- A manual Colab GPU smoke-test checklist.
- A smoke-test results file that records the first real Colab T4 attempt and the current dependency blocker.
- GitHub issue forms and a starter issue plan.
- Guardrails to avoid committing generated datasets, checkpoints, model weights, or secrets.

What does not exist yet:

- Real teacher-model calls.
- A verified Colab GPU training and comparison run.
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

Current skeleton constraints:

- **Input:** `.txt` and `.md` notes only.
- **Teacher:** deterministic local mock teacher.
- **Dataset:** JSONL rows with `instruction`, `response`, and `source_chunk_id`.
- **Training:** optional short TRL/PEFT LoRA entry point; skipped by default and still needs Colab GPU verification.
- **Comparison:** optional base-vs-adapter answer comparison after training; skipped by default and still needs Colab GPU verification.
- **Export:** placeholder only; no GGUF or local runtime output yet.

Planned v0 constraints after the skeleton:

- **Teacher:** one real open-source teacher path, remote only if that improves beginner reliability and is labeled clearly.
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
5. Generate deterministic mock question-answer examples.
6. Preview and save the JSONL dataset in the notebook runtime.
7. Show an optional short training plan that stays skipped by default.
8. If training runs, compare one base-model answer with one trained-adapter answer.
9. Show clear placeholders for export and local running.

The current notebook is [`notebooks/opendistillation_v0_demo.ipynb`](notebooks/opendistillation_v0_demo.ipynb). Its default path runs without GPU, package installs, model downloads, paid APIs, or training. When opened from GitHub in Colab, the setup cell clones this repository before importing local helpers. The optional training cell requires a Colab GPU runtime and extra Hugging Face packages. Use [`docs/colab-smoke-test-checklist.md`](docs/colab-smoke-test-checklist.md) and record results in [`docs/colab-smoke-test-results.md`](docs/colab-smoke-test-results.md) before calling that GPU path verified.

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
    opendistillation_v0_demo.ipynb  # runnable v0 skeleton notebook
  src/
    opendistillation/               # prototype helper package
```

## Development Direction

The current skeleton covers the safe first slice of the notes / school model:

1. Load and validate `.txt` or `.md`.
2. Preview the notes.
3. Chunk text with stable IDs.
4. Generate deterministic mock QA examples.
5. Validate and serialize the JSONL dataset.
6. Prepare a short optional LoRA fine-tuning request from those rows.
7. Prepare an optional before/after comparison from the first generated question.

The next implementation work is to smoke-test the optional training and comparison cells in Colab and choose one real teacher-generation path for the notes model while keeping the mock teacher as a safe fallback. GGUF/local export comes after that.

Future personal model types should reuse the same broad workflow, but they should not be implemented until the notes model path works.

## Contributing

The project is not ready for broad contributions yet. The best early contributions are small and concrete:

- Clarify the README or beginner docs.
- Test the planned Colab notes-model flow.
- Research the most reliable teacher/student defaults.
- Help keep v0 small enough to ship.

Use the issue forms in `.github/ISSUE_TEMPLATE/` when this repository is published.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
