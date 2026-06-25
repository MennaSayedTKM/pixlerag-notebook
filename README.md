# PixleRAG Notebook

**PixleRAG** is a visual document retrieval and question-answering system that works entirely from page images — no text extraction, no OCR, no parsing. It embeds document pages as images, retrieves the most relevant ones using FAISS, and uses GPT-4o vision to read and answer questions directly from the visual content.

This makes it effective for documents where text extraction fails or loses meaning: scanned PDFs, charts, figures, tables, invoices, forms, and mixed-layout reports.

This repository contains a self-contained Google Colab version of PixleRAG — the embedding model runs directly on Colab's GPU, so no local GPU or server setup is required.

---

## Why PixleRAG?

Traditional RAG systems extract text from documents before indexing. This breaks down when:

- The document is a **scanned PDF** with no selectable text
- The answer lives inside a **chart, table, or figure**
- The **layout matters** (invoices, forms, certificates)
- **Copy-pasting loses meaning** (multi-column reports, mixed content)

PixleRAG sidesteps all of this by treating every page as an image from the start. The embedding model understands both visual and textual content, and GPT-4o reads the final pages at high resolution to produce the answer.

---

## Use Cases

| Use case | Why PixleRAG works |
|---|---|
| Scanned PDFs | No OCR needed — pages are embedded as images |
| Charts & figures | GPT-4o reads visual data directly |
| Invoices & forms | Preserves layout and spatial structure |
| Technical reports | Retrieves the exact page with the relevant diagram |
| Mixed-content documents | Text and visuals treated uniformly |
| evaluation | Compare retrieved answers against ground truth |

---

## Pipeline

```
PDF → page images → embed (Qwen3-VL on Colab GPU) → FAISS + MMR → rerank (gpt-4o / Jina) → crop → answer
```

1. **Ingest** — PDF pages are rendered at 300 DPI and embedded using `Qwen3-VL-Embedding-2B` running on Colab's GPU
2. **Retrieve** — Query is embedded and matched against the FAISS index; MMR diversifies results to avoid duplicate pages
3. **Rerank** — GPT-4o (or Jina multimodal reranker) promotes the most relevant pages
4. **Crop** — If the question references a figure, GPT-4o locates and zooms into the relevant region
5. **Answer** — GPT-4o reads the final page(s) at high resolution and synthesises a precise answer

---

## Setup

### 1. API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `OPENAI_API_KEY` | Yes | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `JINA_API_KEY` | Optional | [jina.ai/reranker](https://jina.ai/reranker) (free tier available) |

> If `JINA_API_KEY` is left empty, the pipeline falls back to gpt-4o reranking.

### 2. Upload to Google Drive

Upload these files to `MyDrive/PixleRAG_notebbok/` in your Google Drive:

- `requirements.txt`
- `config.yaml`
- `.env`
- `pixlerag_notebook.ipynb`

The following sample PDFs are included for testing:

- `ocr_evaluation_test.pdf` — test document used for evaluation queries
- `ground_truth.pdf` — ground truth reference document for validating answers

### 3. Run the Notebook

Open `pixlerag_notebook.ipynb` in Google Colab, set the runtime to **T4 GPU**, and run cells **top to bottom**. The Drive mount must complete before the config is loaded.

---

## Usage

```python
# Ingest a document
ingest_file("ocr_evaluation_test.pdf")

# Ask a question
answer, results = ask("What is the recommended minimum scan DPI?")
print(answer)

# Reference a figure explicitly
answer, results = ask("In Figure 1, what does the bar chart show?")
print(answer)

# Search only (no answer synthesis)
results = search("character error rate comparison")
for r in results:
    print(f"page {r.page}  score={r.score:.4f}  {r.source}")
```

---

## Configuration

Edit `config.yaml` to tune behaviour without touching code:

| Setting | Default | Description |
|---------|---------|-------------|
| `reranker` | `gpt-4o` | Reranker model (`gpt-4o` or `jina-reranker-v2-base-multimodal`) |
| `top_k` | `5` | Pages returned after MMR |
| `mmr_lambda` | `0.4` | MMR diversity weight (0 = pure diversity, 1 = pure relevance) |
| `rerank_top_n` | `4` | Pages kept after reranking for answer synthesis |
| `pdf_dpi` | `300` | PDF rendering resolution |
| `max_tile_px` | `1344` | Max image side length sent to embedding model |
| `answer_model` | `gpt-4o` | OpenAI model for answer synthesis |
| `crop_min_px` | `900` | Minimum width of zoomed figure crop |

---

## Requirements

- Google Colab with GPU runtime (T4 or better)
- Python packages listed in `requirements.txt`
- OpenAI API key
