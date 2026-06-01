# Notebooks

This directory will hold the Colab-first OpenDistillation demo.

Planned notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The first notebook should prove one beginner-readable flow:

> Upload a `.txt` or `.md` document, preview and chunk it, generate training examples, fine-tune a small student model, compare before/after behavior, and show the path to local running.

## Notebook Rules

- One main path, not a menu of advanced options.
- Small defaults.
- Clear output after every major step.
- Honest labels for anything not implemented yet.
- No committed generated datasets, checkpoints, model weights, or secrets.

## First Skeleton Target

Before real training is added, the notebook should run top to bottom on CPU and show:

- Introduction and status warning.
- Upload and validation for `.txt` and `.md`.
- Text preview.
- Chunk preview.
- Placeholder sections for teacher generation, student fine-tuning, comparison, and export.
