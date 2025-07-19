# LILA — Fractal Cognitive Engine

LILA is a local AI system that builds and maintains a memory structure using your own data. It doesn't rely on cloud APIs, and it learns from interaction over time.

This is not a chatbot wrapper. It's a self-organizing memory framework built around a quantized large language model (like Mistral-7B), designed to evolve through recursive abstraction, feedback injection, and direct querying.

---

## What it does

- Reads your `.txt`, `.md`, `.py`, `.json`, or `.log` files.
- Chunks the content into small blocks.
- Generates embeddings using a local transformer model.
- Clusters those embeddings into conceptual groups.
- Builds a multi-level hierarchy (fractal memory).
- Lets you ask questions and get relevant responses based on that structure.
- If the answer is good, it saves that memory and integrates it into the structure.

---

## How it learns

Every time you have a meaningful interaction with the system (question + answer), LILA:
- Encodes that interaction as a new chunk.
- Adds it to the base memory (Level 0).
- Rebuilds the higher-level concepts after a few interactions.

This simulates memory formation and integration. The system literally grows as you use it.

---

## How to use it

1. Clone the repo and install dependencies:

pip install sentence-transformers scikit-learn numpy matplotlib torch transformers bitsandbytes accelerate nltk keyboard

2. Launch the app:

python lila.py

3. Use the menu to:
- Train on a folder of documents
- Chat with the system (Lila)
- Query the memory manually
- See how it interprets a question (path explanation)
- Save or load memory state
- Integrate pending memory updates

---

## File overview

- `lila.py`: Main app
- `lila_brain.pkl`: Serialized memory (save/load)
- `lila_log.jsonl`: Log of all interactions (time, question, answer, memory path)
- Any folder of `.txt`, `.md`, `.py`, etc. files: training data

---

## Memory structure

- **Level 0**: All raw embeddings (file chunks + feedback)
- **Levels 1-N**: KMeans cluster centroids, forming abstraction layers
- Labels are generated from frequent words in each cluster
- Queries start at the top level and zoom in through the hierarchy to find the most relevant raw memory

---

## Technical requirements

- GPU with ~10GB VRAM for Mistral-7B (quantized)
- Works offline, no internet required
- Uses PyTorch + Transformers + BitsAndBytes for quantization
- No OpenAI or external API keys needed

---

## Why this exists

This is an experiment in building a memory system that:
- Can live on your own machine
- Can grow over time
- Can learn from you directly
- Can explain why it thinks the way it does

It’s not polished or optimized for production, but the architecture is solid and extensible. It runs locally, stores memories persistently, and evolves over time. The goal is to make something closer to a thinking system — not just another wrapper around a language model.

---

## Things you can do with it

- Train it on your writing or notes
- Chat with it and observe how its memory grows
- Use it to compress and explore your data
- Prototype cognitive agents that retain history
- Experiment with recursive self-training and memory consolidation

---

## What's missing

- No GUI (yet)
- No true long-term planning
- No safety filters
- No evaluation or scoring beyond a basic threshold
- Memory can grow large over time if not pruned

---

## Status

Still experimental. Stable enough to work. Designed to be hacked, extended, and broken in interesting ways.

---

## Author

Created by Yan Desbiens (AI Warlord). Built from scratch, tested on local machines. This is what AI research looks like when you don’t have a lab — just time, compute, and curiosity.

---

## License

MIT
