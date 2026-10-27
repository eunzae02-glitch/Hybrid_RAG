"""Weighted child-rank fusion with one score per independent parent context."""

from src.index_repository import IndexRepository
from src.ranking import best_per_parent
from src.search_bm25 import group_bm25_chunks, rank_bm25_chunks
from src.search_vector import group_vector_chunks, rank_vector_chunks


def fuse_child_ranks(bm25, vector, w_bm25=0.3, w_vec=0.7):
    if w_bm25 < 0 or w_vec < 0 or w_bm25 + w_vec <= 0:
        raise ValueError("검색 가중치는 0 이상이며 합이 0보다 커야 합니다.")
    bm = {str(item["chunk"]["chunk_id"]): (rank, item) for rank, item in enumerate(bm25, start=1)}
    vec = {str(item["chunk"]["chunk_id"]): (rank, item) for rank, item in enumerate(vector, start=1)}
    keys = (set(bm) if w_bm25 else set()) | (set(vec) if w_vec else set())
    results = []
    for key in sorted(keys):
        b_rank, b_item = bm.get(key, (None, None))
        v_rank, v_item = vec.get(key, (None, None))
        score = (w_bm25 / (60 + b_rank) if b_rank else 0) + (w_vec / (60 + v_rank) if v_rank else 0)
        if score <= 0:
            continue
        results.append({"chunk": (b_item or v_item)["chunk"], "rrf_score": score, "bm25_rank": b_rank, "bm25_score": b_item["score"] if b_item else 0, "vector_rank": v_rank, "vector_sim": v_item["similarity"] if v_item else 0})
    return sorted(results, key=lambda item: (-item["rrf_score"], str(item["chunk"]["chunk_id"])))


def group_hybrid_chunks(items, top_k):
    groups = {}
    for item in best_per_parent(items, "rrf_score"):
        groups.setdefault(item["chunk"]["file_name"], []).append(item)
    papers = [{"file_name": name, "max_rrf_score": items[0]["rrf_score"], "sum_rrf_score": sum(item["rrf_score"] for item in items[:3]), "unique_parent_count": len(items), "matched_chunks": items} for name, items in groups.items()]
    return sorted(papers, key=lambda paper: (-paper["sum_rrf_score"], paper["file_name"]))[:top_k]


def search_hybrid_all(query, top_k=3, w_bm25=0.3, w_vec=0.7):
    snapshot = IndexRepository().load()
    if not snapshot.children:
        return [], [], [], [], 0
    bm25, synonyms, _ = rank_bm25_chunks(query, snapshot.children)
    vector = rank_vector_chunks(query, snapshot)
    fused = fuse_child_ranks(bm25, vector, w_bm25, w_vec)
    papers = group_hybrid_chunks(fused, top_k)
    for paper in papers:
        paper["index_revision"] = snapshot.revision
    return group_bm25_chunks(bm25, top_k), group_vector_chunks(vector, top_k), papers, synonyms, len(snapshot.children)
