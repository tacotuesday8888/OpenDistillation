# Notebooks

This directory will hold the Colab-first OpenDistillation demo.

Current notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The current skeleton proves the first safe part of the beginner-readable flow:

> Upload or load a `.txt` or `.md` document, preview and chunk it, generate deterministic mock training examples, and show clear placeholders for future training and export.

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
- Mock teacher generation.
- Placeholder sections for student fine-tuning, comparison, and export.

Later real engines should plug in through the interfaces described in `docs/engine-integration-points.md`.
