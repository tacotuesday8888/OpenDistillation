# First Demo Flow

This is the intended first demo flow. It is a product spec, not implemented code.

## Goal

A user should be able to open a Colab notebook and understand the whole OpenDistillation promise:

> My document became training data, trained a small model, and moved toward local running.

## User Journey

1. Open the Colab notebook from GitHub.
2. Install dependencies.
3. Upload a `.txt` or `.md` file.
4. Preview the uploaded text.
5. Chunk the document into short passages.
6. Generate question-answer pairs from those passages.
7. Preview and optionally download the generated dataset.
8. Select a recommended small student model.
9. Run a short training or fine-tuning job.
10. Ask the base model and trained model the same question.
11. Export the trained output or show the exact next command needed for export.
12. Show local run instructions.

## Success Criteria

- The notebook can be understood by a beginner.
- The user sees a concrete generated dataset.
- The user sees a concrete model behavior comparison.
- The path to local running is explicit.
- The notebook does not require the user to understand distillation theory first.

## First Demo Constraints

- Only `.txt` and `.md` input.
- Only one or two recommended student models.
- Only one teacher path in the default notebook.
- No PDF parsing.
- No web app.
- No Mac app.
- No accounts.
- No billing.

## Open Questions For Implementation

- Which open-source teacher path is most reliable in Colab?
- Which student model gives the best balance of training speed and visible improvement?
- Which export path is realistic in v0 versus documented as a follow-up?
