# Engine Integration Points

The v0 skeleton is designed so the notebook flow can stay stable while real engines are added later.

Current flow:

```text
TXT/MD file
  -> load_text_document()
  -> chunk_text()
  -> TeacherEngine.generate()
  -> validate_dataset() / rows_to_jsonl()
  -> future training engine
  -> future export engine
```

## Text Loading And Chunking

Current helpers:

- `load_text_document(filename, content)`
- `chunk_text(text, max_chars=700)`

These produce a `LoadedTextDocument` and ordered `TextChunk` objects with stable IDs. Later ingestion work can add PDF, DOCX, folders, or web pages by producing the same plain text and chunk objects, but those formats are outside v0.

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

Any real teacher engine must declare whether it sends user text to a remote endpoint.

## Dataset Validation

Current helpers:

- `validate_dataset_row(row)`
- `validate_dataset(rows)`
- `rows_to_jsonl(rows)`

Training engines should consume the validated schema from `docs/dataset-schema.md`. If a later backend needs a different internal format, convert from this schema at the boundary instead of changing the notebook flow.

## Future Training Engines

Training is not implemented in the skeleton.

The intended future boundary is:

```text
validated JSONL rows -> training engine -> adapter/model output
```

Current interface placeholders:

- `TrainingRequest`
- `TrainingResult`
- `TrainingEngine.train(request)`

Candidate open-source engines can include:

- Hugging Face Transformers for model loading and training primitives.
- PEFT/LoRA or QLoRA for small adapter training.
- TRL if it simplifies supervised fine-tuning.
- Unsloth if it gives a simpler or faster Colab path.

The first real training goal should choose one backend after checking current official docs and doing a small Colab smoke test.

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
