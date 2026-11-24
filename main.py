import numpy as np
import ollama
from pypdf import PdfReader
from elasticsearch import Elasticsearch,helpers

PDF_PATH = "./assets/demo rag check.pdf"
EMBED_MODEL = "embeddinggemma-2:latest"
CHAT_MODEL = "phi3"
DIMENSIONS = 256  # Compact 256d vectors for fast in-memory search
IS_INMEMORY = False  # Set to False to use Elasticsearch instead of in-memory search
INDEX_NAME = "leave_policy"  # Elasticsearch index name
client = None
doc_vectors = None
chunks = None

if not IS_INMEMORY:
    client = Elasticsearch("http://localhost:9200", request_timeout=30, max_retries=10, retry_on_timeout=True)


# 2. Create the Elasticsearch index with a dense_vector field (run once)
def create_index_if_missing():
    # 1. Create the vector index (run once)
    if not client.indices.exists(index=INDEX_NAME):
        # index_body = {
        #     "settings": {"index": {"knn": True}},
        #     "mappings": {
        #         "properties": {
        #             "text": {"type": "text"},
        #             "source": {"type": "keyword"},
        #             "chunk_id": {"type": "integer"},
        #             "embedding": {
        #                "type": "dense_vector",
        #                 "dims": DIMENSIONS,
        #                 "index": True,
        #                 "similarity": "cosine"
        #             },
        #         }
        #     },
        # }
        # client.indices.create(index=INDEX_NAME, body=index_body)
        # print(f"Created Elasticsearch index '{INDEX_NAME}'.")
        client.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": DIMENSIONS,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )
        print(f"Created Elasticsearch index '{INDEX_NAME}'.")

# 2. Index PDF chunks into OpenSearch
def index_pdf(path: str):
    reader = PdfReader(path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = [full_text[i : i + 800].strip() for i in range(0, len(full_text), 650)]
    chunks = [c for c in chunks if c]

    vectors = ollama.embed(model=EMBED_MODEL, input=chunks, dimensions=DIMENSIONS).embeddings

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": f"{path}-{idx}",  # Prevents duplicates if you re-run
            "_source": {"text": chunk, "source": path, "embedding": vec, "chunk_id": idx,},
        }
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    helpers.bulk(client, actions, refresh=True)

# 1. Read PDF and split into overlapping text chunks
def load_pdf_chunks_into_memory(path: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    global doc_vectors
    reader = PdfReader(path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    # 2. Embed all chunks once and keep them in RAM (NumPy array)
    print(f"Reading {PDF_PATH} and generating embeddings...")
    embed_response = ollama.embed(
        model=EMBED_MODEL,
        input=chunks,
        dimensions=DIMENSIONS,
    )    
    doc_vectors = np.array(embed_response.embeddings, dtype=np.float32)
    print(f"Indexed {len(chunks)} chunks into RAM ({doc_vectors.nbytes / 1024:.1f} KB).\n")
    return chunks

# 3. Search RAM + Ask phi3
def ask_policy_in_memory(question: str, top_k: int = 3) -> str:
    global doc_vectors, chunks
    # Embed the question using the same model & dimensions
    q_resp = ollama.embed(model=EMBED_MODEL, input=question, dimensions=DIMENSIONS)
    q_vec = np.array(q_resp.embeddings[0], dtype=np.float32)

    # Cosine similarity via dot product (Ollama vectors are already normalized)
    scores = doc_vectors @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Grab the original text of the top matching chunks
    context = "\n\n---\n\n".join(chunks[i] for i in top_indices)

    prompt = f"""Answer the user's question using ONLY the leave policy excerpts below.
If the answer is not in the excerpts, say "I cannot find that in the leave policy."

Leave Policy Excerpts:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

# 3. Search Elasticsearch + Ask phi3
def ask_policy(question: str, top_k: int = 3) -> str:
    q_vec = ollama.embed(model=EMBED_MODEL, input=question, dimensions=DIMENSIONS).embeddings[0]

    # search_body = {
    #     "size": top_k,
    #     "_source": ["text", "source", "chunk_id"],
    #     "query": {
    #         "knn": {
    #             "embedding": {
    #                 "vector": q_vec,
    #                 "k": top_k,
    #                 "num_candidates": 20
    #             }
    #         }
    #     },
    # }
    # hits = client.search(index=INDEX_NAME, body=search_body)["hits"]["hits"]
    response = client.search(
        index=INDEX_NAME,
        source=["text", "source", "chunk_id"],
        knn={
            "field": "embedding",
            "query_vector": q_vec,
            "k": top_k,
            "num_candidates": 20,
        },
    )
    hits = response["hits"]["hits"]
    context = "\n\n---\n\n".join(hit["_source"]["text"] for hit in hits)

    prompt = f"""Answer ONLY using the leave policy excerpts below.
If not found, say "I cannot find that in the leave policy."

Excerpts:
{context}

Question: {question}
Answer:"""

    res = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]


if IS_INMEMORY:
    chunks = load_pdf_chunks_into_memory(PDF_PATH)
else:
     # Index the PDF only if the index is currently empty
    create_index_if_missing()
    doc_count = client.count(index=INDEX_NAME)["count"]
    if doc_count == 0:
        index_pdf(PDF_PATH)
    else:
        print(f"Found {doc_count} chunks already stored in Elasticsearch. Skipping re-indexing.\n")

# 4. Interactive Q&A loop
print("Ready! Ask questions about your leave policy (type 'exit' to quit).\n")

while True:
    user_q = input("You: ").strip()
    if not user_q or user_q.lower() in {"exit", "quit"}:
        break
    if IS_INMEMORY:
        print(f"\nPhi-3: {ask_policy_in_memory(user_q)}\n")
    else:
        print(f"\nPhi-3: {ask_policy(user_q)}\n")