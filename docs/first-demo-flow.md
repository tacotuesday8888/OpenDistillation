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
- A warning that this is an early prototype.
- Runtime guidance: CPU is enough for the default path; GPU is needed if the user opts into the local real teacher or optional fine-tuning cells.

### Step 2: Runtime Setup

The notebook uses the local helper package for the current prototype stage.

Expected output:

- A visible note that the default path uses standard-library Python and local helpers.
- In fresh Colab runtimes, a repository clone into `/content/OpenDistillation` before local helper imports.
- A short note that the optional real teacher and optional training paths need the bounded Hugging Face package set plus Colab's existing GPU `torch`.
- An explicit `INSTALL_TRAINING_DEPS = False` default so local users do not install anything by accident.
- The exact optional install command for Colab GPU users, without upgrading Colab's preinstalled `torch`.

### Step 3: Upload Or Load A Notes File

The notebook loads `examples/sample-notes.md` by default so a beginner can run the first demo in Colab without a file picker. The user can set `USE_SAMPLE_NOTES = False` to upload one `.txt` or `.md` notes file instead.

Validation rules:

- Reject unsupported extensions.
- Reject empty files.
- Show a beginner-readable warning if the file is too short.
- Keep the sample or uploaded content local to the notebook runtime. The optional Qwen teacher downloads model weights from Hugging Face but does not send notes text to a paid or proprietary remote API.

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

### Step 5: Generate Training Examples

The notebook uses a deterministic local mock teacher by default so it can run without model downloads, API calls, GPU, or remote text transfer. The user can set `RUN_REAL_TEACHER = True` in a Colab GPU runtime to use `Qwen/Qwen2.5-1.5B-Instruct` as a local open-source teacher.

Default behavior:

- Generate a small number of examples per chunk.
- Keep the output schema simple.
- Prefer readability over pretending the examples are production quality.
- Label whether the teacher engine sends text to a remote endpoint.
- Keep `RUN_REAL_TEACHER = False` unless the user has installed the optional Hugging Face packages and wants to test the real teacher path.

Expected output:

- A preview of generated examples.
- A JSONL dataset saved in the notebook runtime temp directory.
- A Colab download helper when running in Colab.
- A clear label for `mock-local-teacher` or `huggingface-local-teacher`.
- Plain-language failure messages for missing dependencies, model download/load failures, CUDA or memory failures, and invalid generated rows.

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
- Run a readiness check before training that reports missing optional packages and whether a CUDA GPU is available.
- Save any adapter output under `outputs/`, which is ignored by git.

Expected output:

- A clear training plan.
- A clear "training skipped" message while `RUN_TRAINING = False`.
- Plain-language setup messages for missing packages, no GPU, missing adapter output, or likely GPU memory failures.
- A reminder that Colab GPU verification is still required.

### Step 8: Before/After Comparison

The notebook compares one generated question against the base model and the trained LoRA adapter after optional training.

Default behavior:

- Do not load a model while `RUN_TRAINING = False`.
- Skip comparison when no adapter exists.
- Use the first generated dataset question.
- Show the generated reference answer, base-model answer, and trained-adapter answer when training has run.
- Label the comparison as a qualitative sanity check, not a benchmark.

Expected output:

- A clear "comparison skipped" message while training is skipped.
- A clear comparison plan after training creates an adapter.
- A side-by-side style text output for question, reference answer, base answer, and trained-adapter answer.
- Plain-language setup messages if adapter loading or generation fails.

### Manual Colab Smoke Test

The optional training and comparison path should be called verified only after `docs/colab-smoke-test-checklist.md` is completed.

The checklist records:

- Optional package install success.
- GPU name from the runtime check.
- Model download status.
- Training start and adapter output path.
- Before/after comparison output.
- Runtime and any memory failure.
- Confirmation that generated artifacts stay out of git.

### Step 9: Export Placeholder

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
- The user sees uploaded or sample notes become training examples.
- The user sees where the real teacher opt-in, optional training, before/after comparison, export, and local-run pieces fit.
- All unimplemented pieces are labeled honestly.
- The demo does not imply coding, writing, work, or phone models already exist.

## Failure Cases To Handle

- Unsupported file type.
- Empty or tiny file.
- Too few chunks.
- Teacher generation failure.
- Invalid generated dataset row.
- No-GPU or out-of-memory failures for optional real teacher or training paths.
- Missing optional Hugging Face packages.
- Missing trained adapter before comparison.
- Future export path unavailable.

## V0 Boundaries

Included now:

- `.txt` and `.md` notes loading.
- Plain text preview.
- Simple chunking.
- JSONL dataset schema helpers.
- Local deterministic mock teacher, enabled by default.
- Optional local Qwen real teacher, disabled by default with `RUN_REAL_TEACHER = False`.
- Optional TRL/PEFT LoRA training entry point, skipped by default.
- Optional before/after comparison entry point, skipped by default.
- Runtime readiness checks and manual Colab smoke-test checklist.
- Export placeholder.

Planned later in v0:

- Teacher output quality hardening after the first one-row real-teacher smoke test.
- Colab output/logging resilience for long optional cells.
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
