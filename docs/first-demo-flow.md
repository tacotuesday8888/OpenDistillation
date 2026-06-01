# First Demo Flow

This is the exact v0 user flow for the first Colab prototype. It is a product and implementation spec, not implemented code.

## Demo Goal

A beginner should be able to open one Colab notebook and understand this story:

> My document became training examples, those examples fine-tuned a small model, and I can see how to run the result locally.

The first demo should be small, slow if necessary, and honest. It should not pretend to be production training.

## Default User Input

- One `.txt` or `.md` file.
- Recommended sample: `examples/sample-notes.md`.
- Minimum useful size: roughly 500-2,000 words.
- Unsupported in v0: PDF, DOCX, web pages, images, folders, private drives, and databases.

## Step-By-Step Notebook Flow

### Step 1: Open Notebook

The user opens `notebooks/opendistillation_v0_demo.ipynb` from GitHub in Colab.

Expected output:

- A short explanation of the demo.
- A warning that this is an early prototype.
- Runtime guidance: CPU is enough for the skeleton; GPU is needed for real fine-tuning.

### Step 2: Install Dependencies

The notebook installs only the packages needed for the current prototype stage.

Expected output:

- A visible dependency list.
- A short note that package versions may change while the prototype is being validated.

### Step 3: Upload A Text File

The user uploads one `.txt` or `.md` file.

Validation rules:

- Reject unsupported extensions.
- Reject empty files.
- Show a beginner-readable error if the file is too short.
- Keep the uploaded content local to the notebook runtime unless the teacher path explicitly sends prompts to a remote endpoint.

Expected output:

- File name.
- Character count.
- Approximate word count.
- First 1,000 characters.

### Step 4: Chunk The Document

The notebook splits the text into short chunks.

Default behavior:

- Prefer paragraph boundaries.
- Keep chunks small enough for the chosen teacher prompt.
- Preserve chunk order.
- Drop empty chunks.

Expected output:

- Number of chunks.
- Preview of the first 3 chunks.
- Clear warning if there are too few chunks for training.

### Step 5: Generate Training Examples

The teacher path creates question-answer examples from chunks. The current skeleton uses a deterministic local mock teacher so the notebook can run without model downloads, API calls, GPU, or remote text transfer.

Default behavior:

- Generate a small number of examples per chunk.
- Keep the output schema simple.
- Prefer quality and readability over quantity.
- Label any remote teacher call clearly before it runs.

Expected output:

- A table preview of generated examples.
- A JSONL dataset saved in the notebook runtime temp directory.
- A download link for the dataset.
- A clear label showing whether the teacher engine sends text to a remote endpoint.

Initial JSONL shape:

```json
{"instruction":"Question about the user's document","response":"Answer grounded in the document","source_chunk_id":"chunk-0001"}
```

The schema is documented in `docs/dataset-schema.md`.

### Step 6: Review Dataset

The user sees the dataset before training.

Expected output:

- Count of training examples.
- First 5 examples.
- Simple warnings for very short answers, empty fields, or duplicate questions.

### Step 7: Load Student Model

The notebook loads one recommended small student model.

Default behavior:

- Choose one model around 0.5B-1.5B parameters during implementation.
- Avoid a confusing model picker in v0.
- Explain memory and runtime limits plainly.

Expected output:

- Student model name.
- Estimated hardware requirement.
- Whether GPU is required for the next step.

### Step 8: Run Short Fine-Tuning

The notebook runs a short supervised fine-tuning job.

Default behavior:

- Small batch.
- Small step count.
- Visible progress logs.
- Save adapter/output artifacts only inside the Colab runtime unless the user downloads them.

Expected output:

- Training starts successfully.
- Training finishes or fails with an understandable message.
- Output path inside the runtime.

### Step 9: Compare Before And After

The notebook asks the base model and trained model the same question based on the uploaded document.

Expected output:

- The prompt used for comparison.
- Base model answer.
- Trained model answer.
- Plain-language note that this is a sanity check, not a benchmark.

### Step 10: Save Or Export

The notebook saves the trained output.

Default behavior:

- Save adapter output first if that is the reliable path.
- Attempt GGUF export only if it is stable enough for v0.
- If GGUF export is not implemented, show the exact planned export command and explain what remains unverified.

Expected output:

- Path to saved output.
- Download instruction.
- GGUF/export status.

### Step 11: Run Locally

The notebook shows local run instructions.

Expected output:

- llama.cpp and/or Ollama-style command if export is available.
- If export is not available yet, a clear "not implemented in this prototype" note with the next implementation issue.

## Success Criteria

- The notebook tells one complete story.
- The user sees their uploaded text become training examples.
- The user sees a before/after model response.
- Local running is either demonstrated or documented with an exact next command.
- All unimplemented pieces are labeled honestly.

## Failure Cases To Handle

- Unsupported file type.
- Empty or tiny file.
- Too few chunks.
- Teacher generation failure.
- Invalid generated dataset row.
- No GPU available for training.
- Out-of-memory during training.
- Export path unavailable.

## V0 Boundaries

Included:

- `.txt` and `.md` upload.
- Plain text preview.
- Simple chunking.
- Question-answer dataset generation.
- One student model path.
- One short training path.
- One before/after comparison.
- Local-run guidance.

Current skeleton:

- Implements TXT/MD loading and validation.
- Implements simple chunking.
- Implements the JSONL dataset schema helpers.
- Implements a local deterministic mock teacher.
- Shows training and export placeholders only.

Excluded:

- PDF/DOCX parsing.
- Multi-file projects.
- Web UI.
- SaaS.
- Accounts.
- Billing.
- Cloud job management.
- Long training runs.
- Benchmark claims.
