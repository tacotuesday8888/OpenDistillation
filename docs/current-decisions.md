# Current Decisions

This file records decisions that should not be reopened without a concrete reason.

## Decided

- OpenDistillation is an open-source productized workflow, not a closed SaaS first and not a research-only framework.
- The first public promise is: "Upload docs. Distill a tiny local model. Run it locally."
- The first audience is students, indie developers, AI tinkerers, and technical beginners.
- The first real user experience is Colab-first.
- The local CLI comes later as a thin reproducible wrapper.
- v0 accepts `.txt` and `.md` only.
- v0 uses one default teacher path, not a menu of teachers.
- v0 targets one recommended student model around 0.5B-1.5B parameters.
- The default training method is response distillation / supervised fine-tuning.
- Logits distillation is a later experimental track only if technically feasible.
- The first output should point toward local use through GGUF, llama.cpp, and/or Ollama-style instructions.
- No generated datasets, model weights, checkpoints, API keys, `.env` files, or local machine config should be committed.
- The project uses the Apache-2.0 license.
- The v0 skeleton uses a deterministic local `MockTeacherEngine` before any real teacher-model path is chosen.
- The notebook should call small helper interfaces so real engines can be plugged in later without changing the user flow.

## Not Decided Yet

- Exact teacher model or hosted teacher path.
- Exact student model.
- Exact fine-tuning backend.
- Exact dataset schema fields beyond the required v0 `instruction`, `response`, and `source_chunk_id`.
- Whether GGUF export is implemented in v0 or documented as the immediate next command.
- GitHub repository visibility and remote URL.

## Working Recommendation

Use this v0 flow as the default implementation plan:

1. User opens the Colab notebook.
2. User uploads one `.txt` or `.md` file.
3. The notebook validates and previews the text.
4. OpenDistillation chunks the document.
5. The teacher path generates question-answer pairs.
6. The dataset is previewed, saved, and downloadable.
7. A small student model is loaded.
8. A short supervised fine-tuning run starts.
9. The notebook compares base-model and trained-model answers.
10. The notebook saves the output.
11. The notebook exports to GGUF or shows the exact export command and limitation.
12. The user gets local run instructions.
