# TaxAI Research Assistant

An AI-powered tax research tool that answers U.S. federal tax questions with cited answers sourced directly from official IRS publications. Built using RAG (Retrieval-Augmented Generation) architecture.

---

## Live Demo

[🔗 taxi-research-assistant.streamlit.app](https://taxi-research-assistant.streamlit.app)

---

## What It Does

Most AI chatbots answer tax questions from general training data — which can be outdated or hallucinated. TaxAI is different: it only answers from a curated set of official IRS publications, and it always cites its sources.

- Ask any U.S. federal tax question
- Get a precise, cited answer pulled from real IRS documents
- Click the source link to verify it yourself on IRS.gov

---

## Architecture

```
User Question
      │
      ▼
Embed question (SentenceTransformer)
      │
      ▼
Query ChromaDB vector database
      │
      ▼
Retrieve top 5 relevant IRS document chunks
      │
      ▼
Send chunks + question to Claude API
      │
      ▼
Cited answer returned to Streamlit UI
```

This is a classic **RAG pipeline**:
1. **Ingest** — Downloads 10 IRS PDFs, chunks them into 800-character pieces, embeds them, and stores them in ChromaDB
2. **Retrieve** — Embeds the user's question and finds the most semantically similar chunks
3. **Generate** — Sends the retrieved context to Claude with a strict prompt: only answer from the provided documents

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Claude API (Anthropic) |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | pypdf |
| Language | Python 3 |

---

## IRS Publications Included

- Publication 17 — Your Federal Income Tax
- Publication 463 — Travel, Gift, and Car Expenses
- Publication 502 — Medical and Dental Expenses
- Publication 535 — Business Expenses
- Publication 587 — Business Use of Your Home
- Publication 590-A — Contributions to IRAs
- Publication 590-B — Distributions from IRAs
- Publication 936 — Home Mortgage Interest
- Publication 946 — How To Depreciate Property
- Publication 970 — Tax Benefits for Education

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/devindillon111-png/ai-tax-assistant.git
cd ai-tax-assistant
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```
Get a key at [console.anthropic.com](https://console.anthropic.com)

### 4. Ingest IRS publications
This downloads the PDFs, chunks them, and builds the vector database (~5 minutes):
```bash
python3 ingest.py
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
ai-tax-assistant/
├── app.py            # Streamlit UI
├── rag.py            # Retrieval + Claude API logic
├── ingest.py         # PDF download, chunking, embedding
├── requirements.txt  # Dependencies
├── data/             # Downloaded IRS PDFs (gitignored)
└── db/               # ChromaDB vector store (gitignored)
```

---

## Disclaimer

For research and educational purposes only. Not a substitute for professional tax advice.

---

*Built by Devin Dillon — Accounting Student at the University at Buffalo*
