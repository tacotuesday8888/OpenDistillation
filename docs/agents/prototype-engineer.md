# Agent Brief: Prototype Engineer

## Mission

Build the smallest working OpenDistillation prototype that proves the first personal model journey.

## Core Flow

The first journey is a notes / school model only:

1. Accept `.txt` or `.md` notes input.
2. Chunk the notes.
3. Generate question-answer examples with a safe teacher path.
4. Save and preview the dataset.
5. Prepare or run a small student fine-tune through one efficient backend.
6. Show a simple before/after result when optional training has produced an adapter.
7. Export or document the path to local running later.

## Engineering Principles

- Prefer boring, reliable code.
- Keep Colab and CLI logic shared through Python functions.
- Avoid framework complexity until the notes-model notebook works.
- Use small defaults.
- Make failures understandable.
- Keep future coding, writing, work, and phone flows behind reusable interfaces.

## What To Avoid

- Building a web app.
- Building a Mac app.
- Building a phone app.
- Supporting every document format.
- Supporting every model.
- Building multiple model profiles before the notes model works.
- Building a distributed training system.

## Outputs

- Notebook prototype.
- Minimal Python package.
- Example notes document.
- Example generated dataset format.
- Clear commands for local reproduction.
