# Roadmap

OpenDistillation should move in small phases. Each phase must produce something a user can understand or run.

## Phase 0: GitHub-Ready Foundation

Goal:

> A public visitor can understand the project in under one minute and see a believable first prototype plan.

Deliverables:

- Public README with honest status.
- Product vision and roadmap aligned with the README.
- Exact v0 Colab flow.
- First-demo implementation plan.
- GitHub issue forms and initial issue plan.
- Guardrails against committing secrets, generated datasets, checkpoints, and model artifacts.
- Apache-2.0 license.

Exit criteria:

- Docs agree on the same narrow v0.
- No training pipeline is implied to exist.
- Repo is safe to push once a GitHub remote is ready.

## Phase 1: Colab Skeleton

Goal:

> A user can open the notebook, upload a `.txt` or `.md` file, and see the planned workflow shape without real training yet.

Deliverables:

- `notebooks/opendistillation_v0_demo.ipynb`.
- Upload and text validation cells.
- Text preview and chunk preview.
- Placeholder dataset preview using deterministic sample output.
- Clear labels showing which cells are not implemented yet.

Exit criteria:

- The notebook runs top to bottom without GPU.
- A beginner can see the flow before model work begins.

## Phase 2: Dataset Generation Prototype

Goal:

> Uploaded text becomes a small inspectable JSONL dataset.

Deliverables:

- TXT/MD loader.
- Chunking helper with simple defaults.
- One local deterministic mock teacher-generation path.
- JSONL dataset schema.
- Dataset preview and download step.

Exit criteria:

- A sample input creates a valid dataset.
- The user can inspect examples before training.
- No real teacher model, paid API, model download, GPU, or remote text transfer is required.

## Phase 3: Short Training Path

Goal:

> A small student model can be fine-tuned from the generated dataset in Colab.

Deliverables:

- One recommended student model.
- One efficient supervised fine-tuning path.
- Small default training run.
- Before/after prompt comparison.
- Clear warnings about runtime, GPU, and quality limits.

Exit criteria:

- The notebook demonstrates a visible behavior change on the sample notes.
- The run stays small enough for a beginner Colab demo.

## Phase 4: Local Run Path

Goal:

> The user can save the trained output and follow an explicit path toward local usage.

Deliverables:

- Saved adapter or merged model output.
- GGUF export if practical.
- If GGUF is not practical in v0, exact follow-up command and limitation.
- llama.cpp and/or Ollama-style run instructions.

Exit criteria:

- The notebook does not end at training only.
- The local-running story is honest and testable.

## Phase 5: Thin CLI

Goal:

> The Colab flow can be repeated locally with a small command-line wrapper.

Potential command:

```bash
opendistill run config.yaml
```

The CLI should stay thin. Shared logic should live in the Python package so Colab and local runs do not drift.

## Not In V0

- SaaS.
- Mac app.
- Account system.
- Billing.
- PDF parsing.
- Arbitrary web crawling.
- Large-scale distributed training.
- Full benchmark suite.
- Claims of novel distillation algorithms.
