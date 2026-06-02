# Sample Notes

OpenDistillation is a personal model factory for the AI PC and AI phone era.

The long-term idea is to help people build small personal models for different parts of life: notes and school, coding, writing, work, and eventually phone-local routines.

The first demo should stay much narrower. It should use simple text or Markdown notes, split those notes into chunks, ask a teacher path to create question-answer pairs, and prepare a small notes model from those examples.

The current skeleton uses a deterministic mock teacher. It does not train a real model, download large weights, call paid APIs, or export GGUF files yet.

The final experience should feel simple: a user starts with notes, inspects the generated examples, trains or adapts a small model later, and gets a clear path toward running it locally.
