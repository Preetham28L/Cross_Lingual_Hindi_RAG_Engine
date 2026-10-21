# Cross-Lingual Hindi RAG Engine

A small, reproducible Retrieval-Augmented Generation pipeline that accepts English or Hindi questions and retrieves evidence from a Devanagari Hindi PDF. Every answer includes page, section, chunk ID, source filename, and a retrieval score.

## Architecture

`PDF → PyMuPDF page extraction → NFC/whitespace normalization → paragraph-aware chunks → multilingual embeddings → persistent ChromaDB → top-k retrieval → grounded answer + citations`

## Design choices

- **Embedding model:** `BAAI/bge-m3` by default. It is multilingual and suitable for mapping English queries to Hindi passages. The first run downloads model weights, so plan for network access and several GB of disk/memory depending on the environment.
- **Vector store:** ChromaDB with cosine distance and persistent local storage. It is simple to reproduce and satisfies the assignment's open-source vector DB requirement.
- **Chunking:** paragraph-aware chunks capped around 850 characters, with approximately 120 characters of overlap when a chunk boundary is crossed. Hindi danda (`।`) is preserved and used as a sentence boundary for oversized paragraphs. Metadata includes page, section, chunk ID, source and language.
- **Grounding:** when `GROQ_API_KEY` is configured, the LLM receives only retrieved Hindi passages and is instructed to abstain when the evidence is insufficient. Without an API key, the CLI returns the highest-ranked retrieved passage as an extractive fallback rather than fabricating a generated answer.
- **Citations:** created from metadata returned by ChromaDB, not hard-coded into the answer.
- **Security:** API keys belong in `.env`, which is ignored by Git. Never commit secrets.

## Setup

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `PDF_PATH` to the supplied file location. To enable generated answers, set `GROQ_API_KEY`; otherwise the safe extractive fallback is used.

Place `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` in the project root, then run:

```bash
python rag_engine.py --ingest --reset
python rag_engine.py --query "Which institution did Kalam attend to study aeronautical engineering?"
python rag_engine.py --run-tests
pytest -q
```

The first embedding run may take several minutes and needs model-download access. Subsequent runs reuse local model files and the persistent Chroma collection.

## Evaluation

The six required queries are defined in `TEST_QUERIES` and can be run with `--run-tests`. Inspect the automatically generated source citations and retrieval scores. For a formal retrieval evaluation, compare the top-k retrieved pages/chunks against expected evidence pages in the source PDF. A retrieval score is a cosine-similarity-derived ranking signal, not a probability or faithfulness score.

Do not claim the `<420 ms` TTFT or faithfulness `>=0.85` target has been achieved unless you measure it on the actual deployment hardware and evaluation set.

## Limitations and next steps

1. Table extraction from PDFs can be imperfect; inspect timeline/awards text and add table-specific extraction if required.
2. Section metadata is inferred from numbered headings and may be approximate for chunks near page boundaries.
3. Improve chunking with token-aware sizing and evaluate several chunk sizes/overlaps.
4. Add a small gold evaluation set with expected pages/chunks, Recall@k / MRR, answer faithfulness checks, and latency measurements.
5. Consider a hybrid BM25 + dense retriever and cross-encoder reranker if latency allows.
6. For production, add request logging, timeouts, retries, model caching, and API/UI serving.

## Repository contents

- `rag_engine.py` — ingestion, chunking, embeddings, retrieval, answer generation and CLI.
- `test_rag_engine.py` — basic text-processing tests.
- `requirements.txt` — dependencies.
- `.env.example` — environment configuration template.
