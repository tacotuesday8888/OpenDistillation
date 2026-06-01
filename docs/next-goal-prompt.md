# Recommended Next Goal Prompt

Use this as the next `/goal` after the GitHub-ready documentation foundation is committed:

```text
/goal Create the OpenDistillation v0 Colab notebook skeleton. Work in /Users/langqi/Developer/Projects/OpenDistillation. Add notebooks/opendistillation_v0_demo.ipynb and only the minimal shared Python helpers needed for text upload, validation, preview, and simple TXT/MD chunking. The notebook should run top to bottom on CPU without real model training. Include clear placeholder cells for teacher generation, student fine-tuning, and export, but do not implement real training yet. Verify the notebook can run from a clean local/Jupyter or Colab-compatible environment, update docs if the flow changes, review the diff for secrets and generated artifacts, and commit locally.
```

## Why This Goal

This is the right next step because the project now has enough product clarity. The next useful proof is a notebook that demonstrates the shape of the experience without taking on model training risk immediately.

## Done Means

- `notebooks/opendistillation_v0_demo.ipynb` exists.
- The notebook runs top to bottom without GPU.
- `.txt` and `.md` upload/validation is implemented.
- The user sees file name, character count, word count, and text preview.
- The user sees chunk count and first chunk previews.
- Teacher, training, and export sections are clearly labeled as future implementation.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Real teacher-model calls.
- Real fine-tuning.
- GGUF export implementation.
- SaaS, Mac app, account system, billing, or cloud backend.
