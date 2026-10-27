import os
import json
import re
from rank_bm25 import BM25Okapi
from src.storage import load_corpus_db, DATA_DIR
from src.retrieval_quality import is_searchable_chunk, normalize_search_query

ONTOLOGY_PATH = os.path.join(DATA_DIR, "ontology.json")

def load_ontology() -> dict:
    """data/ontology.json 동의어 사전을 로드합니다."""
    if os.path.exists(ONTOLOGY_PATH):
        try:
            with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def tokenize(text: str) -> list[str]:
    """
    텍스트를 공백 및 특수기호 기준으로 토큰화합니다.
    소문자 변환 및 의미 있는 단어 단위 분할을 수행합니다.
    """
    if not text:
        return []
    separated = re.sub(r"(?<=[a-zA-Z0-9])(?=[가-힣])|(?<=[가-힣])(?=[a-zA-Z0-9])", " ", text)
    clean_text = re.sub(r"[^\w\s]", " ", separated.lower())
    tokens = [t.strip() for t in clean_text.split() if len(t.strip()) > 0]
    return tokens

def expand_query(query: str, ontology: dict) -> tuple[list[str], list[str]]:
    """
    사용자의 질문 쿼리를 ontology.json 동의어 사전을 참조하여 확장합니다.
    
    Returns:
        tuple[list[str], list[str]]: (확장된 토큰 리스트, 감지된 동의어 매핑 리스트)
    """
    base_tokens = tokenize(normalize_search_query(query))
    expanded_set = set(base_tokens)
    synonyms_found = []
    query_lower = query.lower()
    
    for key, synonyms in ontology.items():
        key_match = term_matches_query(key, query_lower)
        synonym_matches = [s for s in synonyms if term_matches_query(s, query_lower)]
        
        if key_match or synonym_matches:
            synonyms_found.append(f"{key} -> {', '.join(synonyms[:3])}")
            expanded_set.update(tokenize(key))
            for syn in synonyms:
                expanded_set.update(tokenize(syn))
                
    return sorted(expanded_set), synonyms_found


def term_matches_query(term: str, query: str) -> bool:
    if re.search(r"[a-zA-Z]", term):
        return re.search(r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])", query, re.IGNORECASE) is not None
    return term.lower() in query.lower()


def build_vector_query(query: str) -> str:
    normalized = normalize_search_query(query)
    expansions = []
    for term, synonyms in load_ontology().items():
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]*", term) or not term_matches_query(term, normalized):
            continue
        english_name = next((s for s in synonyms if re.search(r"[a-zA-Z]", s) and len(s) > len(term) + 5), None)
        if english_name:
            expansions.append(english_name)
    return normalized + (" (" + "; ".join(expansions) + ")" if expansions else "")

def clip_token_frequencies(tokens: list[str], max_repeat: int = 2) -> list[str]:
    """한 청크 내에서 동일한 토큰의 출현 빈도를 최대 max_repeat(기본 2회)까지만 허용하여 단어 도배 뻥튀기를 원천 차단"""
    counts = {}
    clipped = []
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        if counts[t] <= max_repeat:
            clipped.append(t)
    return clipped

def compute_ttr_spam_factor(text: str) -> float:
    """Type-Token Ratio: 고유 단어 수 / 전체 단어 수 비율 기반 키워드 도배(Stuffing) 감점"""
    words = re.findall(r'\w+', text.lower())
    if len(words) < 15:
        return 1.0
    ttr = len(set(words)) / len(words)
    if ttr < 0.40:
        return max(0.15, (ttr / 0.40) ** 2)
    return 1.0


def rank_bm25_chunks(query, corpus):
    corpus = [chunk for chunk in corpus if is_searchable_chunk(chunk)]
    if not corpus:
        return [], [], 0
    expanded, synonyms = expand_query(query, load_ontology())
    tokens = [clip_token_frequencies(tokenize(chunk["text"]), max_repeat=2) for chunk in corpus]
    if not any(tokens):
        return [], synonyms, len(corpus)
    bm25 = BM25Okapi(tokens, k1=1.2, b=0.3)
    scores = bm25.get_scores(expanded)
    ranked = [{"chunk": chunk, "score": float(score) * compute_ttr_spam_factor(chunk["text"])} for chunk, score in zip(corpus, scores) if score > 0]
    ranked.sort(key=lambda item: (-item["score"], str(item["chunk"]["chunk_id"])))
    return ranked, synonyms, len(corpus)


def group_bm25_chunks(ranked, top_k):
    import math
    from src.ranking import best_per_parent
    groups = {}
    for item in best_per_parent(ranked, "score"):
        groups.setdefault(item["chunk"]["file_name"], []).append(item)
    papers = []
    for file_name, items in groups.items():
        maximum = items[0]["score"]
        valid = sum(item["score"] >= maximum * 0.3 for item in items)
        papers.append({"file_name": file_name, "max_score": maximum, "final_score": maximum * (1 + 0.15 * math.log1p(valid)), "valid_chunk_count": valid, "matched_chunks": items})
    return sorted(papers, key=lambda item: (-item["final_score"], item["file_name"]))[:top_k]


def search_bm25(query, top_k=3):
    ranked, synonyms, total = rank_bm25_chunks(query, load_corpus_db())
    return ranked[:top_k], synonyms, total


def search_bm25_grouped(query, top_papers_k=5):
    ranked, synonyms, total = rank_bm25_chunks(query, load_corpus_db())
    return group_bm25_chunks(ranked, top_papers_k), synonyms, total

