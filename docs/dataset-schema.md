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

## Dataset Quality Checks

Schema validation only proves that rows have the right shape. The notebook now also runs deterministic quality checks that do not download models or call any API:

- Row count.
- Number of schema-valid rows.
- Source chunk coverage.
- Missing or unexpected `source_chunk_id` values.
- Duplicate questions.
- Near-duplicate questions within the same source chunk.
- Very short or very long answers.
- Missing required fields.

These checks are dataset-quality signals, not model-quality signals. They help a beginner decide whether the generated rows are worth training on before spending GPU time.

## Current Generators

The notebook uses `MockTeacherEngine`, a deterministic local generator, by default. It is the safe path for local and CPU runs. When a chunk contains simple `Label: value` facts, the mock teacher creates direct note-grounded study rows for those facts. For normal prose or uploaded notes without that shape, it keeps the older excerpt-based fallback.

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

Both generators must return validated rows with `instruction`, `response`, and `source_chunk_id`. The current prompt/templates ask for varied study rows and avoid duplicate or near-duplicate questions in the committed sample-notes path. The real teacher parser rejects invalid JSONL, missing fields, empty fields, and rows with the wrong `source_chunk_id`.

The committed `examples/sample-notes.md` file also has four fixed held-out comparison rows. They use the same schema but different question wording from the mock-teacher training rows. These rows are used only when the committed sample facts are present; uploaded notes fall back to generated comparison questions.

Any future remote teacher path must clearly say when uploaded notes leave the notebook runtime.
