# Engine Integration Points

The v0 prototype is designed so the notebook flow can stay stable while engines are added or hardened.

Current flow:

```text
TXT/MD notes file
  -> load_text_document()
  -> chunk_text()
  -> TeacherEngine.generate()
  -> validate_dataset() / rows_to_jsonl()
  -> analyze_dataset_quality()
  -> extract_fact_ledger() / analyze_fact_quality_gate()
  -> optional SFTLoRATrainingEngine
  -> optional BeforeAfterComparisonEngine
  -> future export engine
```

## Text Loading And Chunking

Current helpers:

- `load_text_document(filename, content)`
- `chunk_text(text, max_chars=700)`

These produce a `LoadedTextDocument` and ordered `TextChunk` objects with stable IDs. The v0 path treats those chunks as notes chunks. Later ingestion work can add coding, writing, work, phone, PDF, DOCX, folders, or web page inputs by producing compatible source text and chunk objects, but those formats are outside v0.

## Teacher Generation

Current interface:

- `TeacherRequest(chunks, examples_per_chunk=4)`
- `TeacherEngine.generate(request)`
- `MockTeacherEngine`
- `HuggingFaceLocalTeacherEngine`
- `RealTeacherConfig`
- `explain_teacher_failure(exc)`

The notebook uses `MockTeacherEngine` by default, which is local and deterministic.

The first optional real teacher engine is `HuggingFaceLocalTeacherEngine`, which loads `Qwen/Qwen2.5-1.5B-Instruct` through Hugging Face Transformers. It sits behind the same `TeacherEngine.generate()` method and returns the same validated JSONL schema.

Both teacher paths now aim for four grounded study-question styles before training: factual recall, explanation, flashcard, and misconception-check. The public row schema remains unchanged.

Current teacher defaults:

- Mock teacher: `mock-local-teacher`, `sends_data_remote = False`, no model download.
- Real teacher: `huggingface-local-teacher`, `sends_data_remote = False`, model weights download from Hugging Face when the user opts in.
- Notebook safety switch: `RUN_REAL_TEACHER = False`.

The real teacher parser accepts JSONL-style model output, validates every row through `validate_dataset()`, and rejects rows that do not use the expected source chunk ID. Failure messages are grouped into dependency, model-load, generation/CUDA-memory, and invalid-output cases.

Later teacher engines can still sit behind the same interface, but v0 should not become a menu of teacher choices before this first notes path is smoke-tested.

Verified locally:

- Mock teacher generation.
- Mock teacher row variety across factual recall, explanation, flashcard, and misconception-check styles.
- Real teacher request construction, JSONL parsing, schema validation, and failure handling with fake no-download dependencies.

Verified once in Colab T4:

- `RUN_REAL_TEACHER = True` style path with `Qwen/Qwen2.5-1.5B-Instruct`, one sample-note chunk, one generated QA row, dataset validation, adapter verification, and before/after comparison.

Still unverified:

- Real teacher output quality beyond the first tiny smoke run.

## Dataset Validation And Quality

Current helpers:

- `validate_dataset_row(row)`
- `validate_dataset(rows)`
- `rows_to_jsonl(rows)`
- `analyze_dataset_quality(rows, expected_chunk_ids=...)`
- `format_dataset_quality_report(report)`

Training engines should consume the validated schema from `docs/dataset-schema.md`. If a later backend needs a different internal format, convert from this schema at the boundary instead of changing the notebook flow.

The quality helper is deterministic and local. It checks row count, valid row count, chunk coverage, missing fields, unexpected source chunk IDs, duplicate questions, near-duplicate questions within the same source chunk, and answer length sanity. It is intentionally not a model benchmark.

The fact-ledger helper is also deterministic and local. It extracts explicit `Label: value` facts and safe bullet/list facts, builds train rows and held-out eval rows with the same public JSONL schema, checks exact and near-duplicate train/eval leakage, and scores exact expected-term hits. It is still a data/eval readiness gate, not proof that a trained model improved.

Hugging Face Evaluate and LightEval were considered for this goal, but not added. They are useful open-source evaluation tools, but this project does not yet have a stable held-out notes benchmark; adding another dependency before that would make the Colab path more fragile without proving more.

## Training Engine

The first real training boundary is:

```text
validated JSONL rows -> SFTLoRATrainingEngine -> PEFT LoRA adapter output
```

Current interface:

- `TrainingRequest`
- `TrainingResult`
- `TrainingEngine.train(request)`
- `SFTLoRAConfig`
- `SFTLoRATrainingEngine`
- `build_training_request(rows, output_dir, config=...)`
- `format_sft_rows(rows)`

Current default:

- Student model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Backend: TRL `SFTTrainer` with PEFT `LoraConfig`.
- Dataset conversion: the v0 `instruction` / `response` rows become conversational prompt/completion examples.
- Output path: `outputs/notes-lora/adapter`, which is ignored by git.
- Default notebook behavior: training is skipped until `RUN_TRAINING = True`.

This choice keeps the first path beginner-readable. TRL provides the supervised fine-tuning wrapper, PEFT keeps the trainable output small, and Qwen2.5-0.5B-Instruct is within the target 0.5B-1.5B student range.

Unsloth and bitsandbytes are not enabled by default. They may become later speed or memory optimization paths after the quality loop can show whether an adapter is actually learning from notes, because they add extra quantization, install, and hardware assumptions.

Verified locally:

- Training request validation.
- Conversion from the v0 dataset schema into TRL prompt/completion rows.
- Training config kwargs and LoRA config kwargs.
- Notebook default path with training skipped.

Verified once in a clean GitHub-opened Colab T4 runtime:

- Optional package installation.
- Student model download.
- The actual `SFTLoRATrainingEngine.train()` call.
- Adapter output creation, memory use, and runtime.
- A 3-step quality smoke from 16 mock-teacher rows created `/content/OpenDistillation-quality-smoke-outputs/notes-lora-quality-smoke/adapter` on a Tesla T4.
- A second 3-step quality smoke after the adapter-disabled comparison fix created `/content/OpenDistillation-quality-smoke-outputs/notes-lora-adapter-disabled/adapter` on a Colab T4 runtime.

Still unverified:

- Adapter quality beyond changed-but-not-improved qualitative smoke checks.

## Before/After Model Quality Engine

The first comparison boundary is:

```text
validated JSONL rows + PEFT adapter output -> BeforeAfterComparisonEngine -> base answers + trained-adapter answers
```

Current interface:

- `BeforeAfterComparisonRequest`
- `BeforeAfterComparisonResult`
- `BeforeAfterComparisonEngine.compare(request)`
- `build_comparison_request(rows, training_result, config=...)`

Current default:

- Question source: up to three validated generated dataset rows, preferring distinct source chunks before reusing a chunk.
- Reference answer: the response from that generated dataset row.
- Base answer: `Qwen/Qwen2.5-0.5B-Instruct` loaded with Transformers and generated through the PEFT model with the LoRA adapter disabled.
- Trained answer: the same base model with the saved PEFT LoRA adapter loaded through `PeftModel.from_pretrained()`.
- Quality signal: a simple lexical reference-overlap value for each base and trained answer.
- Default notebook behavior: comparison is skipped unless optional training creates an adapter.

Verified locally:

- Comparison request validation.
- Bounded multi-question request construction.
- Missing adapter-path handling before any model imports.
- Optional dependency error messages.
- Fake base-vs-adapter generation across multiple questions without downloading models, including a regression test for PEFT adapter-disabled base inference.
- Deterministic reference-overlap scoring.
- Notebook default path with comparison skipped.

Verified in Colab T4 runtime:

- Real base-model generation.
- Real PEFT adapter loading.
- Real trained-adapter generation.
- The multi-question report with three generated dataset questions.
- The adapter-disabled base-answer path after the comparison fix.

Still unverified:

- Whether the answer changes in a useful way in Colab. The first three-question quality smoke produced identical base and trained answers before the adapter-disabled comparison fix; the second produced changed but still generic or hallucinated trained answers.

## Future Export Engines

Export is not implemented in the prototype.

The intended future boundary is:

```text
adapter/model output -> export engine -> local runtime instructions
```

Current interface placeholders:

- `ExportRequest`
- `ExportResult`
- `ExportEngine.export(request)`

Candidate local-run paths can include:

- GGUF export for llama.cpp.
- Ollama-style model packaging if practical.

The README and notebook must not claim local export works until the export path is tested.
