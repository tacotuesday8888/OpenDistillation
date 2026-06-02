# Product Vision

## One-Sentence Idea

OpenDistillation is a personal model factory: an open-source workflow for turning a person's own material into small, focused models for the AI PC and AI phone era.

## Product Promise

The long-term public promise is:

> Build small personal models for the parts of your life.

The v0 promise is narrower:

> Upload TXT/MD notes in Colab, generate training examples, and prepare a tiny local notes model.

This is product packaging around a real workflow. It does not mean OpenDistillation invents a new training algorithm, and it does not mean multiple model types are implemented today.

## Why This Matters

AI PCs and AI phones make local and near-local models more important. A single general assistant will not fit every personal use case. People will want focused models that understand a particular context:

- A notes / school model for study material and personal knowledge.
- A coding model for project patterns and preferred style.
- A writing model for tone, drafts, and editing habits.
- A work model for repeated workflows and role-specific tasks.
- A phone model for lightweight personal routines close to the device.

OpenDistillation should make those models feel buildable by normal technical users, not only ML specialists.

## Target User

The first user is a technical beginner, student, indie developer, researcher, or AI tinkerer who:

- Has notes or task knowledge in plain text.
- Wants a personal model instead of only a retrieval system.
- Can use Colab but does not want to assemble every training step manually.
- Understands that the first model will be small and imperfect.
- Values open-source tools and local ownership.

The first user is not asking for enterprise accounts, dashboards, phone deployment, or a full model-profile system yet.

## Positioning

OpenDistillation should be an open-source productized workflow, not a closed platform and not a research-only toolkit.

It should compete on clarity:

- One narrow beginner path first.
- Honest labels for what is implemented versus planned.
- Clear default choices instead of a maze of options.
- Reusable helper interfaces so future model types can share the same flow.
- Reproducible steps that can later move from Colab to a thin CLI.

## Product Shape

The recommended first shape is:

1. GitHub repository as the public home.
2. Colab notebook as the main first-run experience.
3. Notes / school model as the first model type.
4. Small Python helper package only where it keeps the notebook clean.
5. Thin local CLI later, reusing the same helper code.
6. Export path toward llama.cpp and/or Ollama-style local usage.

Future model profiles can be added only after the first notes-model path works.

## Technical Strategy

Use proven open-source building blocks and make them approachable:

- Hugging Face ecosystem for models and datasets.
- PEFT/LoRA/QLoRA where they keep training small.
- TRL or similar tooling where it simplifies supervised fine-tuning.
- Unsloth or similar acceleration where it improves beginner success.
- llama.cpp/GGUF and Ollama-style instructions for local running.

Exact package and model defaults should be chosen during focused prototype spikes, not guessed in advance.

## Long-Term Direction

If the first notes model works, OpenDistillation can grow into a personal model factory:

- Choose a model type, such as notes, coding, writing, work, or phone.
- Bring source material that fits that model type.
- Generate and inspect training examples.
- Train or adapt a small student model.
- Track simple before/after evaluations.
- Version datasets and model outputs.
- Export multiple local model versions.

The long-term idea is ownership: users should understand where each model came from, what it learned from, and how to improve it.

## Non-Goals

- Multiple model profiles in v0.
- Closed SaaS first.
- Mac app first.
- Phone app first.
- Broad research framework before a demo works.
- PDF and arbitrary ingestion in v0.
- Production cloud training backend in v0.
- Claims of novel low-level optimization without implementation.
