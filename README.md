# OpenDistillation

> Upload docs. Distill a tiny local model. Run it locally.

OpenDistillation is an open-source project for turning a small set of your own notes, docs, or task examples into a small model you can run on your own machine.

The first version is intentionally narrow: a Colab notebook where you upload `.txt` or `.md` files, generate training examples with an open-source teacher model, fine-tune a small student model, and get a clear path toward local running.

This is not a new research algorithm. It is a product-shaped workflow that connects the pieces people already want to use: open-source teacher models, small student models, efficient fine-tuning, GGUF export, and local runtimes such as llama.cpp or Ollama.

## Why This Exists

Fine-tuning and distillation tools are powerful, but the beginner experience is still scattered. OpenDistillation aims to make the personal-model workflow understandable from start to finish:

1. Bring your own text.
2. Turn that text into training examples.
3. Train a small student model.
4. Compare the model before and after training.
5. Export or follow an explicit path toward local use.

The goal is a demo a technical beginner can understand in one sitting, not a giant framework with every possible training method.

## Current Status

OpenDistillation is in foundation mode.

What exists now:

- Public positioning and project docs.
- A scoped v0 Colab flow.
- A first-demo implementation plan.
- GitHub issue forms and a starter issue plan.
- Guardrails to avoid committing generated datasets, checkpoints, model weights, or secrets.

What does not exist yet:

- A real training pipeline.
- A runnable Colab notebook.
- A Python package or CLI.
- A model export implementation.
- A SaaS, account system, Mac app, or cloud backend.

## V0 Scope

The first prototype should prove one complete path:

> A user uploads one `.txt` or `.md` file in Colab, sees generated question-answer training data, fine-tunes a small student model, compares the base and trained model, and gets explicit local-run instructions.

Default constraints:

- **Input:** `.txt` and `.md` only.
- **Teacher:** one open-source teacher path, remote if that improves beginner reliability.
- **Student:** one recommended small model around 0.5B-1.5B parameters.
- **Training:** response distillation / supervised fine-tuning.
- **Export:** GGUF if practical in v0; otherwise document the exact next command and limitation.
- **Local runtime:** llama.cpp and/or Ollama-style instructions.

Out of scope for v0:

- PDF parsing.
- Arbitrary document ingestion.
- Dashboards.
- Accounts or billing.
- Cloud training orchestration.
- Novel distillation algorithm claims.
- Large-scale benchmark suites.

## First Colab Flow

The planned notebook flow is specified in [`docs/first-demo-flow.md`](docs/first-demo-flow.md). In short:

1. Open the notebook from GitHub.
2. Install the pinned prototype dependencies.
3. Upload one `.txt` or `.md` file.
4. Preview and validate the uploaded text.
5. Split it into short chunks.
6. Ask the teacher model to generate question-answer examples.
7. Preview, edit if needed, and save the dataset.
8. Load the recommended student model.
9. Run a short fine-tuning job.
10. Compare base-model and trained-model answers.
11. Save the adapter or model output.
12. Export to GGUF or show the exact export command.
13. Show local run instructions.

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
    first-demo-flow.md              # exact v0 Colab user flow
    first-demo-implementation-plan.md
    github-issue-plan.md
    github-launch-checklist.md
    product-vision.md
    roadmap.md
  examples/
    sample-notes.md                 # tiny input file for the first demo
  notebooks/
    README.md                       # planned notebook location
  src/
    opendistillation/               # future shared Python package
```

## Development Direction

The next implementation milestone is not "build everything." It is:

1. Create a minimal Colab notebook skeleton.
2. Add text upload and validation.
3. Add deterministic chunking for `.txt` and `.md`.
4. Generate a small JSONL dataset from chunks.
5. Add the shortest reliable fine-tuning path.
6. Show before/after behavior.
7. Document the export path honestly.

See [`docs/first-demo-implementation-plan.md`](docs/first-demo-implementation-plan.md) for the concrete build order.

## Contributing

The project is not ready for broad contributions yet. The best early contributions are small and concrete:

- Clarify the README or beginner docs.
- Test the planned Colab flow.
- Research the most reliable teacher/student defaults.
- Help keep v0 small enough to ship.

Use the issue forms in `.github/ISSUE_TEMPLATE/` when this repository is published.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
