# RAG Leave Policy Assistant

This project is a command-line Retrieval-Augmented Generation (RAG) application that answers questions about a leave policy PDF.

## How It Works

- Loads `assets/demo rag check.pdf` using `pypdf`.
- Splits the PDF text into overlapping chunks.
- Generates 256-dimensional embeddings with Ollama's `embeddinggemma-2:latest` model.
- Retrieves the most relevant text chunks using one of two modes:
  - **Elasticsearch mode:** Stores chunks and vector embeddings in a local Elasticsearch index.
  - **In-memory mode:** Keeps the chunks and NumPy embeddings in RAM and uses cosine similarity.
- Sends the retrieved excerpts and the user's question to Ollama's `phi3` chat model.
- Returns answers only from the retrieved leave-policy excerpts and refuses to answer when the information is unavailable.

Set `IS_INMEMORY` in `main.py` to switch between the two retrieval modes.

## Project Files

- `main.py` - PDF loading, chunking, embedding, retrieval, and interactive Q&A.
- `pyproject.toml` - Python package metadata and dependencies.
- `docker/docker-compose.yml` - Local Elasticsearch container configuration.
- `assets/demo rag check.pdf` - Source leave-policy document.

## Requirements

- Python 3.13 or newer
- Ollama
- Docker and Docker Compose, when using Elasticsearch mode

Python dependencies:

- `elasticsearch>=8,<9`
- `numpy>=2.5.5`
- `ollama>=0.6.3`
- `pypdf>=7.1.1`

## Setup

Install Ollama and pull the required models:

```powershell
ollama pull embeddinggemma-2:latest
ollama pull phi3
```

Install the Python project dependencies:

```powershell
uv sync
```

When using Elasticsearch mode, start Elasticsearch:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Elasticsearch will be available at `http://localhost:9200`.

## Run

Start the application:

```powershell
uv run main.py
```

Ask questions in the terminal. Enter `exit` or `quit` to stop.