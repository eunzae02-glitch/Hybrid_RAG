"""Retrieve the same child IDs as BM25, retaining complete page provenance."""

import math

from src.index_repository import IndexRepository
from src.ranking import best_per_parent
from src.retrieval_quality import is_searchable_chunk
from src.search_bm25 import build_vector_query, compute_ttr_spam_factor
from src.vector_store import query_children


def rank_vector_chunks(query, snapshot=None):
    snapshot = snapshot or IndexRepository().load()
    ranked = []
    for child, similarity in query_children(build_vector_query(query), snapshot):
        score = similarity * compute_ttr_spam_factor(child["text"])
        if score > 0 and is_searchable_chunk(child):
            ranked.append({"chunk": child, "similarity": score, "distance": 1.0 - similarity})
    return sorted(ranked, key=lambda item: (-item["similarity"], str(item["chunk"]["chunk_id"])))


def group_vector_chunks(ranked, top_k):
    groups = {}
    for item in best_per_parent(ranked, "similarity"):
        groups.setdefault(item["chunk"]["file_name"], []).append(item)
    papers = []
    for file_name, items in groups.items():
        maximum = items[0]["similarity"]
        valid = sum(item["similarity"] >= maximum * 0.7 for item in items)
        mean = sum(item["similarity"] for item in items[:3]) / min(3, len(items))
        papers.append({"file_name": file_name, "max_similarity": maximum, "final_vector_score": mean * (1 + 0.2 * math.log1p(valid)), "valid_chunk_count": valid, "matched_chunks": items})
    return sorted(papers, key=lambda item: (-item["final_vector_score"], item["file_name"]))[:top_k]


def search_vector_grouped(query, top_papers_k=5):
    snapshot = IndexRepository().load()
    return group_vector_chunks(rank_vector_chunks(query, snapshot), top_papers_k), len(snapshot.children)
