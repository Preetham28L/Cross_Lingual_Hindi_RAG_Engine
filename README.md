# Cross-Lingual Hindi RAG Engine

A small reproducible Retrieval-Augmented Generation pipeline with support for answering questions in English or Hindi using evidence from a Devanagari Hindi PDF. Answers include page, section, chunk, source filename, and retrieval score for each piece of evidence.

## Architecture

PDF → PyMuPDF page extraction → NFC/whitespace normalization → paragraph-aware chunking → multilingual embeddings → persistent ChromaDB → top-k retrieval → grounded answer + citations

## Design choices

Embedding model: BAAI/bge-m3 by default. Multilingual and appropriate for mapping English questions to Hindi passages. First run will download model weights, so make sure to have network connection and several GB of RAM/disk space available depending on your platform.
Vector store: ChromaDB with cosine distance and persistent local storage. Simple to reproduce and satisfies the assignment’s requirement for an open-source vector database.
Chunking: Paragraph-aware chunks of approximately 850 characters with ~120 character overlap when crossing a paragraph boundary. Hindi danda punctuation (।) is used to split overly long paragraphs. Metadata stored with chunks includes page number, section number, chunk id, source filename, and language.
Grounding: When GROQ_API_KEY is configured, the LLM gets only the retrieved Hindi passages as context and is instructed to refrain from making up facts when the evidence is not found. When no API key is given, the CLI will return the highest-ranked retrieved passage as an extractive fallback instead of attempting to generate a synthetic answer.
Citations: Evidence is cited using metadata from the vector store rather than being manually embedded in the answer text.
Security: Keys go in .env which is added to .gitignore. Never commit private keys to version control.

## Setup
Use Python 3.10 or later (3.12 is known to work):

python -m venv .venv

## Windows PowerShell:
.venv\Scripts\Activate.ps1

## macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
export PDF_PATH=... # Set to the directory with the supplied PDF file
export GROQ_API_KEY=... # Set to allow generated answers, otherwise get extractive fallback

## Place Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf in the project root:
python rag_engine.py --ingest --reset
python rag_engine.py --query "Which institution did Kalam attend to study aeronautical engineering?"
python rag_engine.py --run-tests
pytest -q

The first embedding run will take a while and require internet connection to download the model. Subsequent runs will reuse the embedding model and Chroma’s persistent storage.

## Evaluation

The six required queries are specified in TEST_QUERIES in the code, can be run with --run-tests. You should examine the automatically generated source citations and their retrieval scores. For a proper retrieval evaluation, compare the list of retrieved pages/chunks with the ground truth evidence pages for each question in the source PDF. Note that the retrieval score is really a cosine similarity ranking, not a probability, and does not directly indicate faithfulness to the ground truth.
You cannot claim to have achieved the target of <420 ms TTFT or faithfulness >=0.85 without measuring it on the actual target hardware with the full test set.
Limitations and next steps
Table extraction from PDFs might be error prone – review the timeline/awards text and add table-specific processing if needed. Section numbers are guessed heuristically from the markdown headers and might not match the original document exactly for chunks near page boundaries. Consider using a more sophisticated chunking algorithm that uses tokens rather than characters and experiment with chunk sizes and overlaps. It would be good to add a small gold evaluation set with expected pages/chunks, Recall@k, MRR, answer faithfulness, and latency measurements. Perhaps try a hybrid BM25 + dense retriever with cross-encoder reranking if time permits. For productionization, add request logging, timeout handling, retries, model caching, and API/UI serving.

## The repository contains:

rag_engine.py — ingestion, chunking, embeddings, retrieval, answer generation, and CLI
test_rag_engine.py — simple text processing tests
requirements.txt — python dependencies
.env.example — example environment variables file
