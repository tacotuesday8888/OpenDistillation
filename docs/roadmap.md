# OpenDistillation Roadmap

## Phase 0: Project Foundation

- Create the project repository.
- Write the README and product vision.
- Decide initial license.
- Define the first demo flow.
- Prepare a simple public-facing explanation.

## Phase 1: Colab Prototype

Goal:

> A user can upload a `.txt` or `.md` file and run through the full flow in Colab.

Planned flow:

1. Install dependencies.
2. Upload a document.
3. Chunk the document into useful text sections.
4. Ask an open-source teacher model to generate question-answer pairs.
5. Save the generated training dataset.
6. Train a small student model with an efficient backend.
7. Run a small before/after comparison.
8. Export to a local-friendly format where possible.

## Phase 2: CLI Prototype

Goal:

> The same flow can be reproduced locally with one command and a config file.

Potential command:

```bash
opendistill run config.yaml
```

The CLI should stay thin. The real logic should live in the Python package so Colab and CLI share the same workflow.

## Phase 3: Local Model Export

Goal:

> The user can export the trained student model and run it locally.

Focus areas:

- GGUF export path.
- llama.cpp compatibility.
- Ollama instructions if practical.
- Simple local run examples.

## Phase 4: Personal Model Maintenance

Goal:

> Users can keep improving a personal model over time.

Potential features:

- Dataset versioning.
- Model version history.
- Evaluation history.
- Incremental retraining workflow.
- Rollback to previous model versions.

## Not In The First Version

- Full SaaS.
- Mac app.
- PDF and arbitrary document parsing.
- Large-scale multi-node training.
- Full benchmark suite.
- Claims of novel distillation algorithms.
