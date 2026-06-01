# Current Decisions

This file records the decisions made so future agents do not reopen the same questions without a reason.

## Decided

- OpenDistillation should be an open-source productized tool, not a pure framework and not a closed SaaS first.
- The first audience is students, indie developers, AI tinkerers, and technical beginners who want a personal local model.
- The first experience should be Colab-first.
- A CLI should exist later as a thin reproducible wrapper, but it is not the main first-run surface.
- The first demo should use `.txt` and `.md` documents only.
- The first student model target should be around 0.5B-1.5B parameters.
- The default distillation route should be response distillation / SFT.
- Logits distillation can exist as an experimental advanced route later.
- The project should use strong open-source building blocks instead of pretending to invent all low-level training technology.
- The first output should aim toward local use, ideally GGUF and llama.cpp/Ollama-compatible instructions.

## Not Decided Yet

- Exact teacher model.
- Exact student model.
- Exact training backend.
- Whether v0 uses a remote open-source teacher endpoint or a local teacher.
- Exact GGUF export implementation path.
- License. Apache-2.0 or MIT are likely candidates.
- Whether the GitHub repository starts private or public.

## Working Recommendation

Use this v0 flow as the default plan:

1. User opens Colab.
2. User uploads a `.txt` or `.md` file.
3. OpenDistillation chunks the document.
4. Teacher model generates question-answer pairs.
5. The generated dataset is saved and previewed.
6. A small student model is trained or fine-tuned with an efficient backend.
7. The notebook shows a simple before/after comparison.
8. The notebook exports the result or documents the exact export command.
9. The user gets local run instructions.
