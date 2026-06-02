# Current Decisions

This file records decisions that should not be reopened without a concrete reason.

## Decided

- OpenDistillation is an open-source personal model factory for the AI PC and AI phone era.
- The long-term direction supports multiple future personal model types, including notes/school, coding, writing, work, and phone models.
- The v0 product promise is narrower than the long-term vision: start with a notes / school model from TXT/MD input.
- The first real user experience is Colab-first.
- The local CLI comes later as a thin reproducible wrapper.
- v0 accepts `.txt` and `.md` notes only.
- v0 does not implement multiple model profiles.
- v0 uses one default teacher path, not a menu of teachers.
- v0 targets one recommended student model around 0.5B-1.5B parameters when real training begins.
- The default training method is response distillation / supervised fine-tuning.
- Logits distillation is a later experimental track only if technically feasible.
- The first output should point toward local use through GGUF, llama.cpp, and/or Ollama-style instructions.
- No generated datasets, model weights, checkpoints, API keys, `.env` files, or local machine config should be committed.
- The project uses the Apache-2.0 license.
- The v0 skeleton uses a deterministic local `MockTeacherEngine` before any real teacher-model path is chosen.
- The notebook should call small helper interfaces so real engines and future model types can be plugged in later without changing the first flow.

## Not Decided Yet

- Exact teacher model or hosted teacher path.
- Exact student model.
- Exact fine-tuning backend.
- Exact dataset schema fields beyond the required v0 `instruction`, `response`, and `source_chunk_id`.
- Whether GGUF export is implemented in v0 or documented as the immediate next command.
- Which future model profile comes after the notes / school model.
- GitHub repository visibility and remote URL.

## Working Recommendation

Use this notes-model v0 flow as the default implementation plan:

1. User opens the Colab notebook.
2. User uses the sample notes file or uploads one `.txt` or `.md` notes file.
3. The notebook validates and previews the text.
4. OpenDistillation chunks the notes.
5. The mock teacher generates question-answer pairs in the current skeleton.
6. A future real teacher path replaces the mock teacher behind the same interface.
7. The dataset is previewed, saved, and downloadable.
8. A future small student model is loaded.
9. A future short supervised fine-tuning run starts.
10. The notebook compares base-model and trained-model answers.
11. The notebook saves the output.
12. The notebook exports to GGUF or shows the exact export command and limitation.
13. The user gets local run instructions.

Do not build coding, writing, work, or phone model flows until the notes model path works end to end.
