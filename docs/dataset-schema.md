# Dataset Schema

OpenDistillation v0 uses a small JSONL dataset for supervised fine-tuning style examples.

Each line is one JSON object:

```json
{"instruction":"Question about the user's document","response":"Answer grounded in the document","source_chunk_id":"chunk-0001"}
```

## Required Fields

- `instruction` - the question or task the student model should answer.
- `response` - the desired answer, grounded in the source chunk.
- `source_chunk_id` - the chunk ID that produced the example, such as `chunk-0001`.

The helper code validates that all three fields exist and are non-empty strings. Extra fields are not kept in the public v0 schema.

## Current Generator

The current notebook uses `MockTeacherEngine`, a deterministic local generator. It is only for the prototype skeleton.

It does not:

- Call a paid API.
- Download a model.
- Send document text to a remote service.
- Claim to produce high-quality training data.

## Future Teacher Path

Later goals can replace the mock teacher with a real open-source teacher engine while keeping the same dataset shape. The replacement should still return validated rows with `instruction`, `response`, and `source_chunk_id`.

Any remote teacher path must clearly say when uploaded text leaves the notebook runtime.
