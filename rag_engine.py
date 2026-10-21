"""Cross-lingual Hindi RAG pipeline with source-aware citations."""
from __future__ import annotations

import os
import re
import unicodedata
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_DB = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "hindi_reference_chunks"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    section: str
    source: str


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without deleting Hindi punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return (1-based page number, extracted text) for each PDF page."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            text = normalize_text(page.get_text("text", sort=True))
            if text:
                pages.append((idx + 1, text))
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")
    return pages


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, then split oversized blocks on Hindi/Latin sentence ends."""
    blocks = [normalize_text(x) for x in re.split(r"\n\s*\n", text) if normalize_text(x)]
    result: list[str] = []
    for block in blocks:
        if len(block) <= 1100:
            result.append(block)
            continue
        # Keep danda (।), question/exclamation marks and English full stops as boundaries.
        sentences = re.split(r"(?<=[।!?])\s+|(?<=\.)\s+(?=[^\d])", block)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > 850:
                result.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            result.append(current)
    return result


def make_chunks(pdf_path: str | Path, max_chars: int = 850, overlap_chars: int = 120) -> list[Chunk]:
    """Create paragraph-aware chunks with page and section metadata."""
    chunks: list[Chunk] = []
    current_section = "शीर्षक उपलब्ध नहीं"
    for page_num, page_text in extract_pdf_pages(pdf_path):
        lines = [normalize_text(line) for line in page_text.splitlines()]
        # Infer section headings from numbered headings in the supplied study guide.
        for line in lines:
            if re.match(r"^\d{1,2}\s*[·.)]\s*\S", line):
                current_section = line
        paragraphs = split_paragraphs(page_text)
        buffer = ""
        page_sections = current_section
        for para in paragraphs:
            for line in para.splitlines():
                if re.match(r"^\d{1,2}\s*[·.)]\s*\S", line.strip()):
                    page_sections = line.strip()
                    current_section = page_sections
            if len(para) > max_chars:
                pieces = [para[i:i + max_chars] for i in range(0, len(para), max_chars - overlap_chars)]
            else:
                pieces = [para]
            for piece in pieces:
                if buffer and len(buffer) + len(piece) + 2 > max_chars:
                    text = buffer.strip()
                    cid = "p%02d-%s" % (page_num, hashlib.sha1(text.encode("utf-8")).hexdigest()[:10])
                    chunks.append(Chunk(cid, text, page_num, page_sections, Path(pdf_path).name))
                    # Character overlap improves continuity across adjacent paragraphs.
                    buffer = (buffer[-overlap_chars:] + "\n" + piece).strip() if overlap_chars else piece
                else:
                    buffer = (buffer + "\n\n" + piece).strip()
        if buffer.strip():
            text = buffer.strip()
            cid = "p%02d-%s" % (page_num, hashlib.sha1(text.encode("utf-8")).hexdigest()[:10])
            chunks.append(Chunk(cid, text, page_num, page_sections, Path(pdf_path).name))
            buffer = ""
    # IDs are stable for the same page/text, and duplicate IDs are removed.
    unique = {c.chunk_id: c for c in chunks}
    return list(unique.values())


class HindiRAG:
    def __init__(self, model_name: str = DEFAULT_MODEL, db_path: str = DEFAULT_DB):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, pdf_path: str | Path, reset: bool = False) -> int:
        chunks = make_chunks(pdf_path)
        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        # Idempotently upsert the source chunks.
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.model.encode(
                [c.text for c in batch], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ).tolist()
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=vectors,
                metadatas=[{
                    "page": c.page, "section": c.section,
                    "source": c.source, "chunk_id": c.chunk_id,
                    "language": "hi"
                } for c in batch],
            )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            raise RuntimeError("Vector database is empty. Run ingestion first.")
        qvec = self.model.encode([normalize_text(query)], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=qvec,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({
                "text": doc, "metadata": meta,
                "score": round(1.0 - float(distance), 4)
            })
        return hits

    def answer(self, query: str, top_k: int = 4) -> dict:
        hits = self.retrieve(query, top_k)
        context = "\n\n".join(
            f"[{i+1}] page={h['metadata']['page']} section={h['metadata']['section']}\n{h['text']}"
            for i, h in enumerate(hits)
        )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0,
                messages=[
                    {"role": "system", "content":
                     "Answer only from the supplied Hindi evidence. The user may ask in English or Hindi. "
                     "Respond in the user's language. If evidence is insufficient, say so. "
                     "Do not invent facts. Cite evidence as [1], [2] matching context labels."},
                    {"role": "user", "content": f"Question: {query}\n\nHindi source evidence:\n{context}"}
                ],
            )
            answer_text = response.choices[0].message.content.strip()
            mode = "groq_grounded"
        else:
            # Safe no-key mode: return the strongest retrieved source passage rather than hallucinating.
            answer_text = "LLM not configured; strongest retrieved evidence:\n" + hits[0]["text"]
            mode = "extractive_fallback"
        citations = [{
            "page": h["metadata"]["page"],
            "section": h["metadata"]["section"],
            "chunk_id": h["metadata"]["chunk_id"],
            "source": h["metadata"]["source"],
            "score": h["score"],
        } for h in hits]
        return {"query": query, "answer": answer_text, "citations": citations, "mode": mode}


TEST_QUERIES = [
    "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?",
    "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
    "पोखरण-II में कलाम की क्या भूमिका थी?",
    "Which institution did Kalam attend to study aeronautical engineering?",
    "Who co-wrote the autobiography, and in what year was it published?",
    "How and where did Kalam die in 2015?",
]


def format_result(result: dict) -> str:
    lines = [f"Query: {result['query']}", f"Answer: {result['answer']}", "Sources:"]
    for c in result["citations"]:
        lines.append(
            f"- page: {c['page']} | section: {c['section']} | chunk_id: {c['chunk_id']} "
            f"| score: {c['score']} | source: {c['source']}"
        )
    lines.append(f"Mode: {result['mode']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-lingual Hindi RAG CLI")
    parser.add_argument("--pdf", default=os.getenv("PDF_PATH", "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"))
    parser.add_argument("--ingest", action="store_true", help="Extract, embed, and store PDF chunks")
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma collection before ingesting")
    parser.add_argument("--query", help="Ask a Hindi or English question")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--run-tests", action="store_true", help="Run all six required assignment questions")
    args = parser.parse_args()

    rag = HindiRAG()
    if args.ingest:
        count = rag.ingest(args.pdf, reset=args.reset)
        print(f"Ingested/upserted {count} chunks from {args.pdf}")
    elif args.query or args.run_tests:
        queries = TEST_QUERIES if args.run_tests else [args.query]
        for q in queries:
            print("=" * 80)
            print(format_result(rag.answer(q, args.top_k)))
    else:
        parser.print_help()
