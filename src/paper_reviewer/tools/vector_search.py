from crewai.tools import tool
from typing import List, Dict, Any
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
import requests
import json

CHROMA_DIR = "knowledge/data/chroma_lm"
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
LMSTUDIO_MODEL = "lm_studio/liquid/lfm2.5-1.2b"
_vectordb = None


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Call LM Studio embeddings API."""
    data = {"model": LMSTUDIO_MODEL, "input": texts}
    response = requests.post(LMSTUDIO_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return [item["embedding"] for item in result["data"]]


def get_vectordb():
    """Lazy-load Chroma DB with LM Studio embeddings."""
    global _vectordb
    if _vectordb is None:
        from langchain_core.embeddings import Embeddings

        class LMStudioEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return get_embeddings(texts)

            def embed_query(self, text: str) -> List[float]:
                return get_embeddings([text])[0]

        embeddings = LMStudioEmbeddings()
        _vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _vectordb


_last_vector_search_result = None 

@tool("vector_search")
def vector_search(query: str) -> Dict[str, Any]:
    """
    Vector search tool for CrewAI with LM Studio embeddings.
    Returns structured search results and quality metrics.
    """
    top_k = 6
    rerank_top = 3
    batch_size = 2

    vectordb = get_vectordb()
    docs_with_scores = vectordb.similarity_search(query=query, k=top_k)

    if not docs_with_scores:
        return {
            "results": [],
            "metrics": {
                "result_count": 0,
                "top_score": 0.0,
                "avg_score": 0.0,
                "unique_papers": 0,
            },
        }

    # Reranker
    pairs = [(query, d.page_content) for d in docs_with_scores]
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        batch_scores = reranker.predict(
            batch_pairs, convert_to_numpy=True, batch_size=batch_size
        )
        all_scores.extend(batch_scores)

    # Sort by score
    reranked_docs = [
        d
        for _, d in sorted(
            zip(all_scores, docs_with_scores), key=lambda x: x[0], reverse=True
        )
    ]
    sorted_scores = sorted(all_scores, reverse=True)

    # Build results
    results = []
    unique_papers = set()
    for idx, (doc, score) in enumerate(
        zip(reranked_docs[:rerank_top], sorted_scores[:rerank_top])
    ):
        meta = doc.metadata or {}
        arxiv_id = meta.get("arxiv_id", "unknown")
        unique_papers.add(arxiv_id)
        results.append(
            {
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "text": doc.page_content or "",
                "score": float(score),
                "arxiv_id": arxiv_id,
                "paper_title": meta.get("paper_title")
                or meta.get("title")
                or "Unknown Title",
                "year": meta.get("year", 0),
                "section": meta.get("section", "unknown"),
                "subsection": meta.get("subsection", "unknown"),
            }
        )

    # Metrics
    metrics = {
        "result_count": len(results),
        "top_score": float(sorted_scores[0]),
        "avg_score": float(
            sum(sorted_scores[:rerank_top]) / len(sorted_scores[:rerank_top])
        ),
        "unique_papers": len(unique_papers),
    }

    # return {"results": results, "metrics": metrics}
    result = {"results": results, "metrics": metrics}
    global _last_vector_search_result
    _last_vector_search_result = result  
    return result


def get_last_vector_search_result():
    global _last_vector_search_result
    return _last_vector_search_result or {
        "results": [],
        "metrics": {
            "result_count": 0,
            "unique_papers": 0,
            "top_score": 0.0,
            "avg_score": 0.0,
        },
    }
