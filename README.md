# 🧠 AI Tutor for IIT-JEE

> A concept-aware, retrieval-augmented tutor for JEE Advanced. It explains the
> *why* behind a concept, flags the exam-hall traps students fall for, and walks
> through real past-year questions step by step — teaching through their
> verified solutions rather than inventing new ones.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/LLM-Gemini%20API-4285F4.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**Built by [Mohit Chauhan](https://github.com/Mohitcr7)** — Applied AI · RAG · Python

---

## 🎯 Overview

Most "ask-an-LLM" tutors generate answers from the model's memory, which is
risky on an exam where a wrong sign or formula is fatal. This project takes the
opposite approach: it stores **curated, verified exam knowledge** and uses the
LLM to *teach* it, not to recall it.

Two hand-engineered datasets drive everything:

1. **Concept graphs** — the teaching backbone (`Subject → Chapter → concept`,
   each with formulas, common traps, and links to real questions).
2. **A past-year-question (PYQ) corpus** — real JEE Advanced questions, each with
   options, the correct answer, and a **verified worked solution**.

A student's question is answered by *retrieving* the relevant concept and real
questions, then making **one structured LLM call** that produces a concept
explanation, the exam traps, and step-by-step walkthroughs of real problems.

### Core principle

**Verified structured knowledge + retrieval + LLM = a tutor that teaches, not one that guesses.**

The LLM never invents a past-year question, and when it walks through a problem
it teaches *through the provided verified solution* — it does not re-derive the
answer (which would reintroduce the hallucination risk on the exact math you
curated). If a solution looks wrong, it is told to say so rather than silently
change it.

---

## ✨ Key features

- **Concept-first retrieval** — questions are matched to underlying concepts via
  semantic search, not keywords.
- **Teach-through-solution walkthroughs** — real PYQs explained step by step,
  grounded in their verified solutions.
- **Attempt-then-reveal UI** — the question is shown first; the approach hint and
  full walkthrough sit behind expanders, so the student struggles first.
- **Explicit trap awareness** — the mistakes JEE deliberately exploits, per
  concept and per problem.
- **One structured LLM call** — explanation + traps + worked examples returned as
  a validated schema, not free-form text to parse.
- **Self-contained** — local embeddings + on-disk vector store; the only external
  dependency at runtime is the Gemini API.

---

## 🧩 How it works

### Build phase (offline, run once)

```
 data/concept_graph_*.json          Json Files(PYQs)/*_PYQs.json
          │                                   │
          ▼                                   ▼
 build_concept_documents.py         build_pyq_documents.py
          └──────────────┬────────────────────┘
                         ▼
              build_concept_store.py
      embed each document with all-MiniLM-L6-v2 (384-d),
      write to ChromaDB (cosine HNSW)
                         │
                         ▼
        vectorstore/chroma_concepts/
        ├── "jee_concepts"  (concept nodes)
        └── "jee_pyqs"      (questions + verified solutions)
```

Each record is split into an embedded **document** (a compact semantic summary,
what you search against) and **metadata** (the full payload — solutions, traps,
formulas — carried alongside but not embedded). Lists are JSON-encoded because
ChromaDB metadata holds only primitives, and decoded again on retrieval.

### Query phase (runtime, per question)

```
Student question  (+ optional subject filter)
        │
        ▼
retrieve_relevant_concepts ──▶ compress_concepts  (pick the primary concept)
        │
        ▼
collect candidate PYQs = concept's curated links  +  semantic matches
        │
        ▼
build one prompt: concept context + candidate questions WITH verified solutions
        │
        ▼
generate_tutor_response()  ──▶ ONE Gemini call, structured output:
        │                        { concept_explanation, common_traps[],
        │                          worked_examples[] }
        ▼
resolve each worked example's qid → full question (text + solution)
        │
        ▼
Streamlit renders: concept → traps → attempt-then-reveal practice
```

The response is validated against a Pydantic schema, so the app gets typed
objects rather than text to parse.

---

## 🗂️ Data

| Dataset | What it contains | Count |
|---------|------------------|-------|
| **Concept graphs** (`data/`) | Physics + Chemistry + Maths concepts, each with formulas, common traps, and linked question ids | ~657 concepts |
| **PYQ corpus** (`Json Files(PYQs)/`) | JEE Advanced questions with options, answer, verified solution, concept/formula tags, difficulty | 220 questions |

Every question id referenced by the concept graphs resolves to a full question in
the corpus, which is what makes "here's the concept, and here are the real
questions that test it" possible.

---

## 🏗️ Project structure

```
ai_tutor/
├── app.py                          # Streamlit UI (attempt-then-reveal)
├── data/
│   └── concept_graph_*.json        # concept graphs (P / C / M)
├── Json Files(PYQs)/
│   └── {Physics,Chemistry,Maths}/  # PYQ corpus (aggregated *_PYQs.json)
├── embeddings/
│   ├── build_concept_documents.py  # concept graph → documents
│   ├── build_pyq_documents.py      # PYQ corpus → documents
│   └── build_concept_store.py      # build both ChromaDB collections
├── retrieval/
│   └── concept_retriever.py        # semantic retrieval (concepts + PYQs)
├── tutor/
│   ├── tutor_inference.py          # orchestration
│   ├── tutor_prompt_template.py    # single structured prompt
│   └── concept_compressor.py       # rank/focus retrieved concepts
├── llm/
│   └── llm_client.py               # Gemini API + Pydantic output schema
├── vectorstore/                    # persisted store (gitignored; auto-built)
├── requirements.txt
├── .env.example
└── DEPLOY.md
```

---

## 🚀 Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API key
cp .env.example .env          # then edit .env and paste your Gemini API key

# 3. Build the vector store (~1 min; downloads the embedding model once)
python embeddings/build_concept_store.py

# 4. Run
streamlit run app.py
```

> **API key:** the tutor calls the **Google Gemini API**. Get a **free** key at
> [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — the free
> tier is generous but rate-limited (requests per minute/day). If you skip step
> 3, the app builds the store automatically on first question.

### Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | — |
| `TUTOR_MODEL` | Override the model | `gemini-2.5-flash-lite` |

Flash-Lite is the fastest and most free-tier-friendly; step up to
`gemini-2.5-flash` or `gemini-2.5-pro` for deeper reasoning (tighter free-tier
limits).

---

## 📦 Tech stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini API (`gemini-2.5-flash-lite` by default) |
| Structured output | Pydantic schema via Gemini `response_schema` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-d) |
| Vector store | ChromaDB (persistent, cosine HNSW) |
| UI | Streamlit |
| Data format | JSON (concept graphs + PYQ corpus) |

---

## 🚢 Deployment

The app is a single Streamlit process and deploys cleanly to Streamlit Community
Cloud or Hugging Face Spaces. The vector store builds itself on first boot from
the committed source JSON, so it does not need to be committed. See
[DEPLOY.md](DEPLOY.md) for step-by-step instructions and the operational
must-dos (free-tier rate limits, per-user rate limiting) before opening it to
students.

---

## 🔐 Limitations & honesty

- **No evaluation harness yet.** Retrieval quality and explanation correctness
  are not yet measured on a held-out set. This is the top priority next step —
  no performance metrics are claimed until they can be reproduced.
- **Embedding model.** `all-MiniLM-L6-v2` is small and general-purpose; a stronger
  embedder is likely the highest-leverage retrieval upgrade, to be decided by the
  eval set rather than by guess.
- **Corpus coverage.** 220 engineered questions and ~657 concepts across P-C-M;
  broad but not exhaustive.
- **Explanation prose is model-generated.** Worked examples are grounded in
  verified solutions, but the concept explanation itself is the model reasoning
  from the retrieved context — the main surface an eval set should check.
- **No auth or rate limiting yet.** Add per-user throttling before public use.

---

## 🗺️ Roadmap

- [ ] Evaluation set (target concept per query) → measured retrieval + answer quality
- [ ] Stronger embedding model, chosen by eval
- [ ] Per-user rate limiting and simple auth
- [ ] Student progress / concept-mastery tracking
- [ ] Expand the PYQ corpus and concept coverage

---

## 👤 Author

**Mohit Chauhan** — [github.com/Mohitcr7](https://github.com/Mohitcr7)

Interested in Data Science / AI Engineering roles. Built this end to end:
knowledge engineering, the RAG retrieval layer, the structured LLM pipeline,
and the deployment.

---

## 📄 License

MIT — intended for educational use. JEE Advanced questions are sourced from
official IIT-JEE publications. (Add a `LICENSE` file if you publish this.)
