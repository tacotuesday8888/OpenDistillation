# Agent Brief: Prototype Engineer

## Mission

Build the smallest working OpenDistillation prototype that proves the core user journey.

## Core Flow

1. Accept `.txt` or `.md` input.
2. Chunk the document.
3. Generate question-answer examples with an open-source teacher path.
4. Save and preview the dataset.
5. Train or fine-tune a small student model through an efficient backend.
6. Show a simple before/after result.
7. Export or document the path to local running.

## Engineering Principles

- Prefer boring, reliable code.
- Keep Colab and CLI logic shared through Python functions.
- Avoid framework complexity until the notebook works.
- Use small defaults.
- Make failures understandable.

## What To Avoid

- Building a web app.
- Building a Mac app.
- Supporting every document format.
- Supporting every model.
- Building a distributed training system.

## Outputs

- Notebook skeleton.
- Minimal Python package.
- Example input document.
- Example generated dataset format.
- Clear commands for local reproduction.
