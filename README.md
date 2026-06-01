# OpenDistillation

> Fine-tuning has Unsloth. Personal model distillation has OpenDistillation.

OpenDistillation is an open-source project for turning your own documents or task data into a small local model.

The goal is not to be another research-only distillation framework. The goal is to make the full workflow feel simple:

1. Upload your docs.
2. Use an open-source teacher model to generate training data.
3. Train a small student model with efficient open-source tools.
4. Export the result to GGUF.
5. Run your own tiny model locally.

## Current Status

This project is in the planning and bootstrapping stage.

The first milestone is a Colab-first prototype that proves one clear user journey:

> Upload a `.txt` or `.md` file, distill it into a small model, export it, and run it locally.

## Product Direction

OpenDistillation should be an open-source productized tool:

- Open-source core.
- Colab-first beginner experience.
- CLI for reproducible local runs.
- Optional integration with strong open-source backends such as Unsloth, Hugging Face Transformers, PEFT, TRL, llama.cpp, GGUF, and Ollama.

## What This Is Not

- Not a closed SaaS first.
- Not a full Mac app first.
- Not a giant research framework first.
- Not a claim that OpenDistillation invents a new distillation algorithm.

## First Version

The first version should focus on one successful flow:

- Input: `.txt` or `.md` document.
- Teacher: open-source model, preferably remote or optional local.
- Student: small model around 0.5B-1.5B parameters.
- Training: response distillation / SFT as the default path.
- Advanced: experimental logits distillation only where technically feasible.
- Output: GGUF model and instructions for running locally.

## Repository Layout

```text
OpenDistillation/
  README.md
  docs/
    product-vision.md
    roadmap.md
    superpowers/plans/
  notebooks/
  src/
    opendistillation/
```

## License

License is not chosen yet. A permissive license such as Apache-2.0 or MIT is likely the safest default for an open-source developer tool.
