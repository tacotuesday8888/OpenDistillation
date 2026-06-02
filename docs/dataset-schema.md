# Dataset Schema

OpenDistillation v0 uses a small JSONL dataset for supervised fine-tuning style examples.

Each line is one JSON object:

```json
{"instruction":"Question about the user's notes","response":"Answer grounded in the notes","source_chunk_id":"chunk-0001"}
```

## Required Fields

- `instruction` - the question or task the notes model should answer.
- `response` - the desired answer, grounded in the source chunk.
- `source_chunk_id` - the chunk ID that produced the example, such as `chunk-0001`.

The helper code validates that all three fields exist and are non-empty strings. Extra fields are not kept in the public v0 schema.

## Current Generators

The notebook uses `MockTeacherEngine`, a deterministic local generator, by default. It is the safe path for local and CPU runs.

It does not:

- Call a paid API.
- Download a model.
- Send notes text to a remote service.
- Claim to produce high-quality training data.

The notebook also includes an opt-in real teacher path:

- Engine: `HuggingFaceLocalTeacherEngine`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Switch: `RUN_REAL_TEACHER = True`.
- Data behavior: model weights download from Hugging Face; notes text is not sent to a paid or proprietary remote API.

Both generators must return validated rows with `instruction`, `response`, and `source_chunk_id`. The real teacher parser rejects invalid JSONL, missing fields, empty fields, and rows with the wrong `source_chunk_id`.

Any future remote teacher path must clearly say when uploaded notes leave the notebook runtime.
