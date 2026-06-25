# PixleRAG Notebook

A self-contained Google Colab notebook for visual document retrieval and Q&A — no text parsing, everything is image-based.

## Pipeline

```
PDF → page images → embed (Qwen3-VL on Colab GPU) → FAISS + MMR → rerank (gpt-4o / Jina) → crop → answer
```

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

Open `pixlerag_notebook.ipynb` in Google Colab and run cells **top to bottom**. The Drive mount must complete before the config is loaded.

## Usage

```python
# Ingest a document
ingest_file("my_document.pdf")

# Ask a question
answer, results = ask("What is the recommended minimum scan DPI?")
print(answer)

# Search only (no answer synthesis)
results = search("Q4 revenue by product line")
```

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

## Requirements

- Google Colab with GPU runtime (for Qwen3-VL embedding)
- Python packages listed in `requirements.txt`
- OpenAI API key
