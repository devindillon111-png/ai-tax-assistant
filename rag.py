"""
rag.py — Retrieval-Augmented Generation Pipeline
==================================================
This is the core of the system. Given a user's question, it:
  1. Converts the question to an embedding (same model used in ingest.py)
  2. Finds the most relevant chunks from ChromaDB
  3. Sends those chunks + the question to Claude
  4. Returns Claude's answer with source citations

This file is imported by app.py — you don't run it directly.
"""

import os
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# Load ANTHROPIC_API_KEY from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

DB_DIR = Path("db")
COLLECTION_NAME = "irs_docs"
TOP_K = 5            # How many chunks to retrieve per question
MODEL = "claude-opus-4-8"  # Claude model to use


# ─────────────────────────────────────────────────────────
# LOAD MODELS (done once, cached for the app session)
# ─────────────────────────────────────────────────────────

# These are loaded once when the module is imported — not on every question.
# This is important: loading a model takes a few seconds, so we only do it once.
print("Loading embedding model...")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
_chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
_collection = _chroma_client.get_collection(COLLECTION_NAME)

print("Connecting to Claude API...")
_claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("RAG system ready.\n")


# ─────────────────────────────────────────────────────────
# STEP 1: RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────────────────

def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most relevant document chunks for a given question.

    How it works:
    - Convert the question to an embedding (same model as ingest.py)
    - ChromaDB compares that embedding to all stored embeddings
    - Returns the top_k most semantically similar chunks

    This is the "R" in RAG — Retrieval.
    """
    # Embed the question — same model, same vector space
    question_embedding = _embedding_model.encode([question]).tolist()[0]

    # Query ChromaDB for the closest matching chunks
    results = _collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Unpack results into a clean list of dicts
    chunks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "relevance_score": round(1 - dist, 3)  # Convert distance to similarity score
        })

    return chunks


# ─────────────────────────────────────────────────────────
# STEP 2: GENERATE ANSWER WITH CLAUDE
# ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for Claude.
    We include the source filename so Claude can cite it.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline: question → retrieve → generate → return answer + citations.

    Returns a dict with:
      - "answer": Claude's response text
      - "sources": list of source filenames cited
      - "chunks": the raw retrieved chunks (useful for debugging)
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in my knowledge base. Try rephrasing your question.",
            "sources": [],
            "chunks": []
        }

    # Step 2: Build the context from retrieved chunks
    context = format_context(chunks)

    # Step 3: Build the prompt for Claude
    # This is called a "RAG prompt" — we inject retrieved context before the question.
    # Claude is instructed to ONLY use what's in the context, so it doesn't hallucinate.
    system_prompt = """You are a professional tax research assistant with expertise in U.S. federal tax law.
You answer questions strictly based on the IRS publications provided in the context below.
Do not use any knowledge outside of what is provided.

Rules:
- Always cite your sources using the format: [Source N: filename]
- If the context doesn't contain enough information to answer confidently, say so
- Use plain English — avoid jargon unless necessary, and define terms when you use them
- Be specific: quote or closely paraphrase the relevant IRS language when possible
- If there are important caveats or exceptions, mention them"""

    user_message = f"""Here are the relevant sections from IRS publications:

{context}

---

Based ONLY on the IRS publications above, please answer this question:

{question}"""

    # Step 4: Send to Claude
    response = _claude_client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "user", "content": user_message}
        ],
        system=system_prompt
    )

    answer_text = response.content[0].text

    # Extract unique source filenames from the retrieved chunks
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": chunks
    }


# ─────────────────────────────────────────────────────────
# QUICK TEST (run this file directly to test the pipeline)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Can I deduct my home office if I work from home?",
        "What is the standard deduction for 2023?",
        "How does depreciation work for business equipment?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)
        result = answer_question(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
