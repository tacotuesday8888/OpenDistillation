# Product Vision

## One-Sentence Idea

OpenDistillation helps people turn their own documents or task knowledge into a small local model they can understand, run, and keep improving.

## Product Promise

The public promise should stay simple:

> Upload docs. Distill a tiny local model. Run it locally.

That promise is product packaging around a real workflow. It does not mean OpenDistillation invents a new training algorithm. It means the project makes the workflow easier to follow end to end.

## Target User

The first user is a technical beginner, student, indie developer, researcher, or AI tinkerer who:

- Has notes, docs, or task knowledge in plain text.
- Wants a personal model instead of only a retrieval system.
- Can use Colab but does not want to assemble every training step manually.
- Understands that the first model will be small and imperfect.
- Values open-source tools and local ownership.

## Positioning

OpenDistillation should be an open-source productized workflow, not a closed platform and not a research-only toolkit.

It should compete on clarity:

- One narrow beginner path first.
- Honest labels for what is implemented versus planned.
- Clear default choices instead of a maze of options.
- Reproducible steps that can later move from Colab to a thin CLI.

## Product Shape

The recommended first shape is:

1. GitHub repository as the public home.
2. Colab notebook as the main first-run experience.
3. Small Python helper package only where it keeps the notebook clean.
4. Thin local CLI later, reusing the same helper code.
5. Export path toward llama.cpp and/or Ollama-style local usage.

## Technical Strategy

Use proven open-source building blocks and make them approachable:

- Hugging Face ecosystem for models and datasets.
- PEFT/LoRA/QLoRA where they keep training small.
- TRL or similar tooling where it simplifies supervised fine-tuning.
- Unsloth or similar acceleration where it improves beginner success.
- llama.cpp/GGUF and Ollama-style instructions for local running.

Exact package and model defaults should be chosen during the first prototype spike, not guessed in advance.

## Long-Term Direction

If the first demo works, OpenDistillation can grow into a personal model lifecycle tool:

- Add more documents over time.
- Generate and inspect new training examples.
- Retrain or continue training.
- Track simple before/after evaluations.
- Version datasets and model outputs.
- Export multiple local model versions.

The long-term idea is ownership: users should understand where their model came from and how to improve it.

## Non-Goals

- Closed SaaS first.
- Mac app first.
- Broad research framework before a demo works.
- PDF and arbitrary ingestion in v0.
- Production cloud training backend in v0.
- Claims of novel low-level optimization without implementation.
