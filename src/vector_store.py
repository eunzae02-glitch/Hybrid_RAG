"""Explicit, length-checked embeddings for immutable Chroma index revisions."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

from src.chunk_config import MODEL_DIR, model_signature, stable_hash
from src.index_repository import IndexIntegrityError, IndexRepository
from src.token_budget import get_token_budget

COLLECTION_NAME = "paper_children"
_client_lock = RLock()


def embedding_signature(model_hash=None):
    return stable_hash({"model": model_hash or model_signature(), "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    if model.max_seq_length != get_token_budget().limit:
        raise IndexIntegrityError("청커와 임베딩 모델의 토큰 한도가 다릅니다. 재색인 설정을 확인해 주세요.")
    return model


class SafeEmbeddingFunction:
    def __call__(self, texts):
        budget = get_token_budget()
        for text in texts:
            budget.validate(text)
        if not texts:
            return []
        return get_embedding_model().encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embedding_function():
    return SafeEmbeddingFunction()


@lru_cache(maxsize=12)
def _cached_client(path):
    import chromadb
    return chromadb.PersistentClient(path=path)


def _client(path):
    # lru_cache allows concurrent cache misses to construct duplicate clients.
    # Chroma publishes its System before Rust bindings finish starting, so the
    # lock must cover the cache lookup AND construction, not just its factory.
    canonical = os.path.normcase(str(Path(path).resolve()))
    with _client_lock:
        return _cached_client(canonical)


def embedding_hash(child):
    return hashlib.sha256(child["embedding_text"].encode("utf-8")).hexdigest()


def _metadata(child):
    return {
        "child_id": child["child_id"], "chunk_id": str(child["chunk_id"]), "parent_id": child["parent_id"],
        "document_id": child["document_id"], "file_name": child["file_name"],
        "page_num": child["page_num"], "section_title": child["section_title"], "kind": child["kind"],
        "parse_method": child["parse_method"], "token_count": child["token_count"],
        "embedding_hash": embedding_hash(child), "source_spans": json.dumps(child["source_spans"], ensure_ascii=False),
    }


def index_children(path, children, fingerprint, previous=None, embedding_signature=None):
    collection = _client(str(path / "chroma_db")).create_collection(
        name=COLLECTION_NAME, embedding_function=None,
        metadata={"hnsw:space": "cosine", "fingerprint": fingerprint},
    )
    reuse = None
    if previous is not None and not previous.legacy and previous.manifest.get("embedding_signature") == embedding_signature:
        reuse = get_chroma_collection(previous)
    reused = 0
    for start in range(0, len(children), 64):
        batch = children[start:start + 64]
        cached = {}
        if reuse is not None:
            existing = reuse.get(ids=[c["child_id"] for c in batch], include=["metadatas", "embeddings"])
            if existing["ids"]:
                for identity, metadata, vector in zip(existing["ids"], existing["metadatas"], existing["embeddings"]):
                    cached[identity] = (metadata.get("embedding_hash"), vector.tolist() if hasattr(vector, "tolist") else vector)
        missing = [c for c in batch if c["child_id"] not in cached or cached[c["child_id"]][0] != embedding_hash(c)]
        calculated = get_embedding_function()([c["embedding_text"] for c in missing])
        new_vectors = {c["child_id"]: vector for c, vector in zip(missing, calculated)}
        vectors = []
        for child in batch:
            if child["child_id"] in new_vectors:
                vectors.append(new_vectors[child["child_id"]])
            else:
                vectors.append(cached[child["child_id"]][1])
                reused += 1
        collection.upsert(ids=[c["child_id"] for c in batch], documents=[c["text"] for c in batch], metadatas=[_metadata(c) for c in batch], embeddings=vectors)
    validate_vector_records(collection, children)
    return {"count": collection.count(), "reused_embeddings": reused, "new_embeddings": len(children) - reused}


def get_chroma_collection(snapshot=None):
    snapshot = snapshot or IndexRepository().load()
    if snapshot.legacy:
        raise IndexIntegrityError("기존 색인을 토큰 기반 색인으로 재생성해 주세요.")
    return _client(str(snapshot.path / "chroma_db")).get_collection(COLLECTION_NAME, embedding_function=None)


def validate_vector_records(collection, children):
    records = collection.get(include=["metadatas"])
    expected = {c["child_id"]: c for c in children}
    if set(records["ids"]) != set(expected):
        raise IndexIntegrityError("벡터와 검색 청크의 ID가 일치하지 않습니다.")
    for identity, metadata in zip(records["ids"], records["metadatas"]):
        child = expected[identity]
        if any(metadata.get(key) != value for key, value in {"parent_id": child["parent_id"], "page_num": child["page_num"], "embedding_hash": embedding_hash(child)}.items()):
            raise IndexIntegrityError("벡터의 본문 해시 또는 출처 정보가 일치하지 않습니다.")


_verified_revisions = set()


def sync_corpus_to_vector_db(snapshot=None):
    snapshot = snapshot or IndexRepository().load()
    if snapshot.legacy:
        raise IndexIntegrityError("'데이터 현황'에서 토큰 기반 색인을 생성해 주세요.")
    from src.chunk_config import pipeline_signature
    if snapshot.manifest["model_signature"] != model_signature() or snapshot.manifest["pipeline_signature"] != pipeline_signature():
        raise IndexIntegrityError("모델 또는 청킹 코드가 색인 생성 시점과 다릅니다. '데이터 현황'에서 색인을 재생성해 주세요.")
    collection = get_chroma_collection(snapshot)
    if collection.metadata.get("fingerprint") != snapshot.manifest["fingerprint"]:
        raise IndexIntegrityError("벡터 색인의 버전이 본문 데이터와 다릅니다.")
    if collection.count() != len(snapshot.children):
        raise IndexIntegrityError("벡터 개수가 본문 데이터와 다릅니다. 색인을 재생성해 주세요.")
    if snapshot.revision not in _verified_revisions:
        validate_vector_records(collection, snapshot.children)
        _verified_revisions.add(snapshot.revision)
    return collection


def _query_vectors(text):
    budget = get_token_budget()
    if budget.count(text) <= budget.limit:
        return get_embedding_function()([text])[0]
    offsets = budget.offsets(text)
    windows, start = [], 0
    while start < len(offsets):
        end = min(len(offsets), start + budget.limit - 2)
        while budget.count(text[offsets[start][0]:offsets[end - 1][1]]) > budget.limit:
            end -= 1
        windows.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == len(offsets):
            break
        start = max(start + 1, end - 16)
    import numpy as np
    mean = np.mean(get_embedding_function()(windows), axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm else mean).tolist()


def query_children(query, snapshot=None, limit=150):
    snapshot = snapshot or IndexRepository().load()
    collection = sync_corpus_to_vector_db(snapshot)
    if not snapshot.children:
        return []
    results = collection.query(query_embeddings=[_query_vectors(query)], n_results=min(limit, len(snapshot.children)))
    children = snapshot.child_map
    return [(children[identity], max(0.0, 1.0 - float(distance))) for identity, distance in zip(results["ids"][0], results["distances"][0]) if identity in children]
