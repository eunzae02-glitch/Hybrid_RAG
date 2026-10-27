"""Build and validate a complete new index; publishing is a separate step."""

from datetime import datetime, timezone

from src.chunk_config import CHUNKER_VERSION, PARSER_VERSION, ChunkSettings, build_fingerprint, model_signature, pipeline_signature
from src.chunker_smart import build_document_chunks
from src.index_repository import FORMAT_VERSION, IndexRepository, IndexSnapshot, file_digest, write_json
from src.index_validation import validate_documents
from src.token_budget import get_token_budget


def build_snapshot(documents, settings=None, repository=None, previous=None, progress=None):
    settings = settings or ChunkSettings()
    repository = repository or IndexRepository()
    documents = sorted(documents, key=lambda doc: doc["file_name"])
    budget = get_token_budget()
    signature = model_signature()
    fingerprint = build_fingerprint(documents, settings, signature)
    parents, children = [], []
    for index, document in enumerate(documents):
        if progress:
            progress(f"토큰 청킹 {index + 1}/{len(documents)}: {document['file_name']}")
        doc_parents, doc_children = build_document_chunks(document, settings, budget)
        parents.extend(doc_parents)
        children.extend(doc_children)
    report = validate_documents(documents, parents, children, budget, settings)
    revision, path = repository.new_revision()
    for name, value in (("documents", documents), ("parents", parents), ("children", children)):
        write_json(path / f"{name}.json", value)
    if progress:
        progress(f"{len(children)}개 청크 벡터 색인 생성 및 검증")
    from src.vector_store import embedding_signature, index_children
    vector_signature = embedding_signature(signature)
    vectors = index_children(path, children, fingerprint, previous=previous, embedding_signature=vector_signature)
    report["vector_count"] = vectors["count"]
    manifest = {
        "format_version": FORMAT_VERSION, "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint, "model_signature": signature, "embedding_signature": vector_signature, "pipeline_signature": pipeline_signature(),
        "parser_version": PARSER_VERSION, "chunker_version": CHUNKER_VERSION,
        "settings": settings.to_dict(), "validation": report, "embeddings": vectors,
        "files": {f"{name}.json": file_digest(path / f"{name}.json") for name in ("documents", "parents", "children")},
    }
    write_json(path / "manifest.json", manifest)
    return IndexSnapshot(revision, path, manifest, documents, parents, children)


def rebuild_library(settings=None, min_text_len_per_page=50, data_dir=None, progress=None):
    repository = IndexRepository(data_dir) if data_dir is not None else IndexRepository()
    with repository.writer():
        previous = repository.load()
        settings = settings or ChunkSettings.from_dict(previous.manifest.get("settings"))
        from src.parser import extract_document_from_pdf
        documents = []
        for path in sorted((repository.data_dir / "uploaded_pdfs").glob("*.pdf")):
            if progress:
                progress(f"PDF 읽기: {path.name}")
            documents.append(extract_document_from_pdf(path.read_bytes(), path.name, min_text_len_per_page))
        snapshot = build_snapshot(documents, settings, repository, previous=previous, progress=progress)
        repository.activate(snapshot)
        return snapshot
