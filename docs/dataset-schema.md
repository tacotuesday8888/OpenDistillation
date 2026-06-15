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

## Fact-Ledger Sidecar

The public training JSONL stays deliberately small. The fact-ledger path keeps richer quality metadata in an internal sidecar so beginners can still inspect the dataset without learning a larger schema.

For explicit notes like `Label: value` or safe list items like `- Label - value`, the fact ledger stores one structured fact card with:

- `fact_id` - stable ID such as `fact-0001`.
- `source_chunk_id` - the chunk that contained the fact.
- `label` and `value` - the extracted fact.
- `expected_terms` - exact terms that a correct answer must include.
- `fact_kind` - currently `label_value` or `list_pair`.
- `source_hash` - short hash of the source chunk text for traceability.

The train rows and held-out eval rows still use the same public `instruction`, `response`, and `source_chunk_id` shape. The sidecar manifest links each row back to `row_id`, `fact_id`, `split`, `source_hash`, `fact_kind`, `label`, and `expected_terms`. This lets the quality gate check whether training and eval are separated without committing generated datasets or model artifacts.

After the 2026-06-08 0/8 fact-ledger GPU result, the sidecar also records `value` and `row_style`. These are internal diagnostics only. They let OpenDistillation explain whether a row is an exact-value answer-only target, a label/value recall target, a canonical `Label: value` binding target, or a held-out direct-recall eval row, while the public JSONL schema stays unchanged.

Held-out comparison can also use in-memory rows enriched from this sidecar with `row_id`, `fact_id`, `label`, `value`, `row_style`, and `expected_terms`. Those fields are not written to the public JSONL. They keep exact fact-hit scoring attached to the selected fact even when comparison questions are reordered to cover distinct source chunks first.

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
- Fact coverage for train and held-out eval rows.
- Exact and near-duplicate leakage between train questions and eval questions, including token-overlap cases where the words are mostly the same but rearranged.
- Expected-term coverage, so each train/eval response contains the exact fact it is supposed to teach or test. The check ignores case and punctuation, but does not accept partial-word matches.

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

The current notebook also builds a deterministic fact-ledger report from the committed sample notes. The safe default path shows the number of extracted facts, six-row-per-fact train rows, held-out eval rows, train/eval leakage count, expected-term checks, and the first few fact cards before any optional training starts. A passing report means the local data/eval split is safer to test; it does not mean the small model has learned the facts.

The 2026-06-08 Colab T4 fact-ledger smokes used this same public schema for 24 fact-ledger train rows and 8 held-out eval rows. The schema and leakage checks passed in both the pre-fix run and the revised value-first run, but the trained adapter still scored 0/8 exact expected-term hits, the same as the base model. This branch keeps the public schema stable while increasing the local fact-ledger signal to 48 train rows for the same 8 facts. The next GPU experiment should test that exact shape once, with exact fact-hit scoring and failure recorded if changed answers still miss the facts.

Any future remote teacher path must clearly say when uploaded notes leave the notebook runtime.
