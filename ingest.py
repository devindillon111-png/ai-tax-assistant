"""
ingest.py — Document Ingestion Pipeline
========================================
This script does 3 things:
  1. Downloads IRS publications as PDFs
  2. Extracts the text and splits it into chunks
  3. Converts those chunks into embeddings and stores them in ChromaDB

You only need to run this ONCE (or when you want to add new documents).
After that, your vector database is built and ready to answer questions.

Run with: python ingest.py
"""

import os
import requests
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

# IRS publications to download and index.
# Format: { "filename": "IRS download URL" }
# These are real, publicly available IRS PDFs.
IRS_PUBLICATIONS = {
    "pub17.pdf": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
    "pub946.pdf": "https://www.irs.gov/pub/irs-pdf/p946.pdf",
    "pub535.pdf": "https://www.irs.gov/pub/irs-pdf/p535.pdf",
    "pub587.pdf": "https://www.irs.gov/pub/irs-pdf/p587.pdf",
    "pub936.pdf": "https://www.irs.gov/pub/irs-pdf/p936.pdf",
    "pub502.pdf": "https://www.irs.gov/pub/irs-pdf/p502.pdf",
    "pub970.pdf": "https://www.irs.gov/pub/irs-pdf/p970.pdf",
    "pub590a.pdf": "https://www.irs.gov/pub/irs-pdf/p590a.pdf",
    "pub590b.pdf": "https://www.irs.gov/pub/irs-pdf/p590b.pdf",
    "pub463.pdf": "https://www.irs.gov/pub/irs-pdf/p463.pdf",
}
DATA_DIR = Path("data")          # Where we store downloaded PDFs
DB_DIR = Path("db")              # Where ChromaDB stores its files
CHUNK_SIZE = 800                 # Characters per chunk (roughly 150 words)
CHUNK_OVERLAP = 100              # Overlap between chunks to avoid cutting sentences mid-thought
COLLECTION_NAME = "irs_docs"    # Name of our ChromaDB collection


# ─────────────────────────────────────────────────────────
# STEP 1: DOWNLOAD PDFS
# ─────────────────────────────────────────────────────────

def download_pdfs():
    """Download IRS publications if we don't already have them."""
    DATA_DIR.mkdir(exist_ok=True)

    for filename, url in IRS_PUBLICATIONS.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"  ✓ {filename} already downloaded, skipping")
            continue

        print(f"  ↓ Downloading {filename} from IRS.gov...")
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            filepath.write_bytes(response.content)
            print(f"  ✓ Saved {filename} ({len(response.content) // 1024} KB)")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")


# ─────────────────────────────────────────────────────────
# STEP 2: EXTRACT TEXT AND CHUNK IT
# ─────────────────────────────────────────────────────────

def extract_text_from_pdf(filepath: Path) -> str:
    """Pull all text out of a PDF file, skipping any corrupted pages."""
    try:
        reader = PdfReader(filepath, strict=False)
    except Exception as e:
        print(f"     Warning: Could not open {filepath.name}: {e}")
        return ""
    text = ""
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception:
            pass
    return text


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split a long document into overlapping chunks.

    Why chunks? Because:
    - LLMs have context limits — we can't send a 200-page PDF in one message
    - Smaller chunks make retrieval more precise (find exactly what's relevant)
    - Overlap ensures we don't split important sentences in half

    Returns a list of dicts: {"text": "...", "source": "pub17.pdf", "chunk_id": "pub17_0"}
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text_content = text[start:end]

        # Don't create tiny leftover chunks (less than 100 chars)
        if len(chunk_text_content) < 100 and chunk_index > 0:
            break

        chunks.append({
            "text": chunk_text_content,
            "source": source_name,
            "chunk_id": f"{source_name}_{chunk_index}"
        })

        chunk_index += 1
        start = end - CHUNK_OVERLAP  # Overlap: step back slightly before next chunk

    return chunks


def load_all_documents() -> list[dict]:
    """Extract and chunk all downloaded PDFs."""
    all_chunks = []

    for filepath in DATA_DIR.glob("*.pdf"):
        print(f"  → Processing {filepath.name}...")
        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text, filepath.name)
        all_chunks.extend(chunks)
        print(f"     Created {len(chunks)} chunks from {filepath.name}")

    return all_chunks


# ─────────────────────────────────────────────────────────
# STEP 3: EMBED AND STORE IN CHROMADB
# ─────────────────────────────────────────────────────────

def build_vector_database(chunks: list[dict]):
    """
    Convert text chunks into embeddings and store them in ChromaDB.

    What's an embedding?
    - A list of ~384 numbers that represents the *meaning* of a piece of text
    - Similar meaning = similar numbers = close together in vector space
    - This is how we search by meaning instead of exact keywords

    SentenceTransformers gives us a free, local embedding model —
    no API call needed, runs entirely on your machine.
    """
    DB_DIR.mkdir(exist_ok=True)

    print("\n  Loading embedding model (downloads ~90MB on first run)...")
    # 'all-MiniLM-L6-v2' is small, fast, and surprisingly good for this use case
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete existing collection if it exists (so we can re-run cleanly)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  ✓ Cleared existing database")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Process in batches of 100 (more efficient than one-at-a-time)
    batch_size = 100
    total = len(chunks)

    print(f"\n  Embedding and storing {total} chunks...")
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{"source": c["source"]} for c in batch]

        # This is the key step: convert text → numbers (embeddings)
        embeddings = model.encode(texts).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        progress = min(i + batch_size, total)
        print(f"    {progress}/{total} chunks stored...", end="\r")

    print(f"\n  ✓ Done! {total} chunks stored in ChromaDB")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AI Tax Assistant — Document Ingestion Pipeline")
    print("=" * 55)

    print("\n[1/3] Downloading IRS publications...")
    download_pdfs()

    print("\n[2/3] Extracting text and chunking documents...")
    chunks = load_all_documents()

    if not chunks:
        print("\n  ✗ No chunks created — check that PDFs downloaded correctly")
        exit(1)

    print(f"\n  Total chunks across all documents: {len(chunks)}")

    print("\n[3/3] Building vector database...")
    build_vector_database(chunks)

    print("\n" + "=" * 55)
    print("  ✓ Ingestion complete! Your database is ready.")
    print("  Next step: run the app with:  streamlit run app.py")
    print("=" * 55)
