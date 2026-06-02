# Sample Notes

OpenDistillation is a personal model factory for the AI PC and AI phone era.

The long-term idea is to help people build small personal models for different parts of life: notes and school, coding, writing, work, and eventually phone-local routines.

The first demo should stay much narrower. It should use simple text or Markdown notes, split those notes into chunks, ask a teacher path to create question-answer pairs, and prepare a small notes model from those examples.

The current notebook uses a deterministic mock teacher by default. It can show an optional short fine-tuning plan, but it does not download large weights, call paid APIs, run training, or export GGUF files unless the user explicitly opts in where supported.

The final experience should feel simple: a user starts with notes, inspects the generated examples, optionally trains or adapts a small model, and gets a clear path toward running it locally.
