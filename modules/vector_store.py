"""
ChromaDB vector store with BAAI/bge-base-en-v1.5 embeddings (local, no API).

Embedding model : BAAI/bge-base-en-v1.5
Dimensions      : 768
Similarity      : cosine
Persistence     : ./storage/

BGE models achieve best retrieval by prepending a task prefix to queries
(not to stored documents). See: https://huggingface.co/BAAI/bge-base-en-v1.5
"""

import os
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "pdf_chunks"
PERSIST_DIR = "./storage"


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self._encoder = SentenceTransformer(EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=persist_dir)
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self._encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def add_chunks(self, chunks: List[Dict]) -> None:
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"page": c["page"]} for c in chunks]
        self._collection.add(
            ids=ids,
            embeddings=self._embed(texts, is_query=False),
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        q_emb = self._embed([query_text], is_query=True)[0]
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
        )
        return [
            {
                "text": doc,
                "page": meta["page"],
                "score": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    @property
    def count(self) -> int:
        return self._collection.count()
