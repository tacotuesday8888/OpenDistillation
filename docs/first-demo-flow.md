# First Demo Flow

This is the exact v0 user flow for the first Colab prototype. It is a product and implementation spec for the notes / school model, not a broad multi-profile system.

## Demo Goal

A beginner should be able to open one Colab notebook and understand this story:

> My notes became training examples, those examples can optionally start a tiny model fine-tune, and I can see the path toward running that model locally.

The first demo should be small, slow if necessary, and honest. It should not pretend to be production training, a phone app, a coding model, or a complete personal model factory.

## Default User Input

- One `.txt` or `.md` notes file.
- Recommended sample: `examples/sample-notes.md`.
- Minimum useful size: roughly 500-2,000 words.
- Unsupported in v0: PDF, DOCX, web pages, images, folders, private drives, databases, coding repositories, workspaces, phone data, and multi-profile inputs.

## Step-By-Step Notebook Flow

### Step 1: Open Notebook

The user opens `notebooks/opendistillation_v0_demo.ipynb` from GitHub in Colab.

Expected output:

- A short explanation of the notes-model demo.
- A warning that this is an early prototype skeleton.
- Runtime guidance: CPU is enough for the default path; GPU is needed only if the user opts into the optional fine-tuning cell.

### Step 2: Runtime Setup

The notebook uses the local helper package for the current prototype stage.

Expected output:

- A visible note that the skeleton uses standard-library Python and local helpers.
- A short note that optional training needs current Hugging Face training packages.

### Step 3: Upload Or Load A Notes File

The user uploads one `.txt` or `.md` notes file, or uses the sample notes file when running locally.

Validation rules:

- Reject unsupported extensions.
- Reject empty files.
- Show a beginner-readable warning if the file is too short.
- Keep the uploaded content local to the notebook runtime unless a future teacher path explicitly sends prompts to a remote endpoint.

Expected output:

- File name.
- Character count.
- Approximate word count.
- First 1,000 characters.

### Step 4: Chunk The Notes

The notebook splits the text into short chunks.

Default behavior:

- Prefer paragraph boundaries.
- Keep chunks small enough for the chosen teacher prompt.
- Preserve chunk order.
- Drop empty chunks.

Expected output:

- Number of chunks.
- Preview of the first 3 chunks.
- Clear warning if there are too few chunks for optional training.

### Step 5: Generate Mock Training Examples

The current skeleton uses a deterministic local mock teacher so the notebook can run without model downloads, API calls, GPU, or remote text transfer.

Default behavior:

- Generate a small number of examples per chunk.
- Keep the output schema simple.
- Prefer readability over pretending the examples are production quality.
- Label whether the teacher engine sends text to a remote endpoint.

Expected output:

- A preview of generated examples.
- A JSONL dataset saved in the notebook runtime temp directory.
- A Colab download helper when running in Colab.
- A clear label that `MockTeacherEngine` is local and deterministic.

Initial JSONL shape:

```json
{"instruction":"Question about the user's notes","response":"Answer grounded in the notes","source_chunk_id":"chunk-0001"}
```

The schema is documented in `docs/dataset-schema.md`.

### Step 6: Review Dataset

The user sees the dataset before optional training.

Expected output:

- Count of training examples.
- First 5 examples.
- Simple validation of required fields.

### Step 7: Optional Training Entry Point

The notebook shows a bounded real student fine-tuning path, but keeps it skipped by default.

Default behavior:

- Do not load a model.
- Do not use GPU.
- Do not fine-tune.
- Build a training request from the validated dataset rows.
- Show a plan for `Qwen/Qwen2.5-0.5B-Instruct` with TRL `SFTTrainer` and PEFT LoRA.
- Require `RUN_TRAINING = True` before model download or adapter training begins.
- Save any adapter output under `outputs/`, which is ignored by git.

Expected output:

- A clear training plan.
- A clear "training skipped" message while `RUN_TRAINING = False`.
- A reminder that Colab GPU verification is still required.

### Step 8: Export Placeholder

The notebook shows where GGUF/local-runtime export will plug in later.

Default behavior:

- Do not create GGUF files.
- Do not claim llama.cpp or Ollama local running works yet.
- Explain that export comes after real training output exists.

Expected output:

- A clear "export skipped" message.
- A reminder that no model artifacts or local runtime files are created.

## Success Criteria

- The notebook tells one honest notes-model story.
- The user sees uploaded or sample notes become mock training examples.
- The user sees where a real teacher, optional training, export, and local-run pieces fit.
- All unimplemented pieces are labeled honestly.
- The demo does not imply coding, writing, work, or phone models already exist.

## Failure Cases To Handle

- Unsupported file type.
- Empty or tiny file.
- Too few chunks.
- Teacher generation failure.
- Invalid generated dataset row.
- Future no-GPU or out-of-memory training failures.
- Future export path unavailable.

## V0 Boundaries

Included now:

- `.txt` and `.md` notes loading.
- Plain text preview.
- Simple chunking.
- JSONL dataset schema helpers.
- Local deterministic mock teacher.
- Optional TRL/PEFT LoRA training entry point, skipped by default.
- Export placeholder.

Planned later in v0:

- One real teacher path for notes.
- Colab GPU smoke test for the optional short training path.
- One before/after comparison.
- Local-run guidance.

Excluded:

- Multiple model profiles.
- Coding model.
- Writing model.
- Work model.
- Phone model.
- PDF/DOCX parsing.
- Multi-file projects.
- Web UI.
- SaaS.
- Accounts.
- Billing.
- Cloud job management.
- Long training runs.
- Benchmark claims.
