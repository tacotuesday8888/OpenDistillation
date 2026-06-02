# Notebooks

This directory holds the Colab-first OpenDistillation demo.

Current notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The current skeleton proves the first safe part of the notes / school model flow:

> Load a `.txt` or `.md` notes file, preview and chunk it, generate deterministic mock training examples, and show clear placeholders for future training and export.

## Notebook Rules

- One notes-model path, not a menu of personal model profiles.
- Small defaults.
- Clear output after every major step.
- Honest labels for anything not implemented yet.
- No committed generated datasets, checkpoints, model weights, or secrets.

## First Skeleton Target

Before real training is added, the notebook should run top to bottom on CPU and show:

- Introduction and status warning.
- Upload/loading and validation for `.txt` and `.md` notes.
- Text preview.
- Chunk preview.
- Mock teacher generation.
- Placeholder sections for student fine-tuning, comparison, and export.

Later real engines should plug in through the interfaces described in `docs/engine-integration-points.md`.
