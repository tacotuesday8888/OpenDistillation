# Engine Integration Points

The v0 skeleton is designed so the notebook flow can stay stable while real engines are added later.

Current flow:

```text
TXT/MD notes file
  -> load_text_document()
  -> chunk_text()
  -> TeacherEngine.generate()
  -> validate_dataset() / rows_to_jsonl()
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

- `TeacherRequest(chunks, examples_per_chunk=2)`
- `TeacherEngine.generate(request)`
- `MockTeacherEngine`

The notebook currently uses `MockTeacherEngine`, which is local and deterministic.

Later real teacher engines can sit behind the same `TeacherEngine.generate()` method. Possible implementations:

- A local open-source teacher model through Hugging Face Transformers.
- A hosted open-source teacher endpoint, if that is more reliable for beginners.
- A batch generation job that still returns the same JSONL rows.

Any real teacher engine must declare whether it sends user notes to a remote endpoint.

## Dataset Validation

Current helpers:

- `validate_dataset_row(row)`
- `validate_dataset(rows)`
- `rows_to_jsonl(rows)`

Training engines should consume the validated schema from `docs/dataset-schema.md`. If a later backend needs a different internal format, convert from this schema at the boundary instead of changing the notebook flow.

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

Unsloth and bitsandbytes are not enabled by default. They may become later optimization paths after the plain TRL/PEFT adapter run is verified, because they add extra quantization, install, and hardware assumptions.

Verified locally:

- Training request validation.
- Conversion from the v0 dataset schema into TRL prompt/completion rows.
- Training config kwargs and LoRA config kwargs.
- Notebook default path with training skipped.

Requires Colab/GPU verification:

- Optional package installation.
- Student model download.
- The actual `SFTLoRATrainingEngine.train()` call.
- Adapter quality, memory use, and runtime.

## Before/After Comparison Engine

The first comparison boundary is:

```text
validated JSONL rows + PEFT adapter output -> BeforeAfterComparisonEngine -> base answer + trained-adapter answer
```

Current interface:

- `BeforeAfterComparisonRequest`
- `BeforeAfterComparisonResult`
- `BeforeAfterComparisonEngine.compare(request)`
- `build_comparison_request(rows, training_result, config=...)`

Current default:

- Question source: the first validated generated dataset row.
- Reference answer: the response from that generated dataset row.
- Base answer: `Qwen/Qwen2.5-0.5B-Instruct` loaded with Transformers.
- Trained answer: the same base model with the saved PEFT LoRA adapter loaded through `PeftModel.from_pretrained()`.
- Default notebook behavior: comparison is skipped unless optional training creates an adapter.

Verified locally:

- Comparison request validation.
- Missing adapter-path handling before any model imports.
- Optional dependency error messages.
- Fake base-vs-adapter generation without downloading models.
- Notebook default path with comparison skipped.

Requires Colab/GPU verification:

- Real base-model generation.
- Real PEFT adapter loading.
- Real trained-adapter generation.
- Whether the answer changes in a useful way after the short training run.

## Future Export Engines

Export is not implemented in the skeleton.

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
