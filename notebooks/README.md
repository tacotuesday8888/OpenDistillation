# Notebooks

This directory holds the Colab-first OpenDistillation demo.

Current notebook:

```text
notebooks/opendistillation_v0_demo.ipynb
```

The current skeleton proves the first safe part of the notes / school model flow:

> Load a `.txt` or `.md` notes file, preview and chunk it, generate deterministic mock training examples, and prepare an optional short TRL/PEFT LoRA training run that stays skipped by default.

## Notebook Rules

- One notes-model path, not a menu of personal model profiles.
- Small defaults.
- Clear output after every major step.
- Honest labels for anything not implemented yet.
- No committed generated datasets, checkpoints, model weights, or secrets.

## Current Notebook Target

With training skipped by default, the notebook should run top to bottom on CPU and show:

- Introduction and status warning.
- Upload/loading and validation for `.txt` and `.md` notes.
- Text preview.
- Chunk preview.
- Mock teacher generation.
- Optional student fine-tuning section with `RUN_TRAINING = False` by default.
- Placeholder sections for comparison and export.

Later real engines should plug in through the interfaces described in `docs/engine-integration-points.md`.

The optional training section is not part of the CPU smoke path. It requires a Colab GPU runtime and installing the Hugging Face training packages listed in the notebook. It saves adapters under `outputs/`, which is ignored by git.
