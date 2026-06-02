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

> Upload TXT/MD notes in Colab, generate mock training examples, and prepare the path toward a tiny local notes model.

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
- Helper interfaces for future teacher, training, and export engines.
- A first-demo implementation plan.
- GitHub issue forms and a starter issue plan.
- Guardrails to avoid committing generated datasets, checkpoints, model weights, or secrets.

What does not exist yet:

- Real teacher-model calls.
- Real model training.
- Multiple personal model profiles.
- Coding, writing, work, or phone model flows.
- A CLI.
- A model export implementation.
- A SaaS, account system, Mac app, or cloud backend.

## V0 Scope

The v0 prototype is only the first personal model type:

> A notes / school model from one `.txt` or `.md` file.

The first prototype should prove one complete path:

> A user uploads one `.txt` or `.md` notes file in Colab, sees generated question-answer training data, fine-tunes a small student model later, compares the base and trained model later, and gets explicit local-run instructions later.

Current skeleton constraints:

- **Input:** `.txt` and `.md` notes only.
- **Teacher:** deterministic local mock teacher.
- **Dataset:** JSONL rows with `instruction`, `response`, and `source_chunk_id`.
- **Training:** placeholder only; no real fine-tuning yet.
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
7. Show clear placeholders for real teacher generation, training, comparison, export, and local running.

The current skeleton notebook is [`notebooks/opendistillation_v0_demo.ipynb`](notebooks/opendistillation_v0_demo.ipynb). It runs without GPU, model downloads, paid APIs, or real training.

## Repository Map

```text
OpenDistillation/
  README.md                         # public project page
  START_HERE.md                     # quick orientation for future work
  AGENTS.md                         # working agreements for coding agents
  .github/
    ISSUE_TEMPLATE/                 # structured GitHub issue forms
  docs/
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

The next implementation milestone is to choose one real teacher-generation path for the notes model while keeping the mock teacher as a safe fallback. Real fine-tuning, before/after comparison, and GGUF/local export come after that.

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
