"""Library operations publish complete index revisions and preserve originals on failure."""

import hashlib
from pathlib import Path
import uuid

from src.chunk_config import ChunkSettings, pipeline_signature
from src.index_repository import DATA_DIR as ROOT_DATA_DIR, IndexRepository, replace_file

DATA_DIR = str(ROOT_DATA_DIR)
UPLOAD_DIR = str(ROOT_DATA_DIR / "uploaded_pdfs")
CORPUS_JSON_PATH = str(ROOT_DATA_DIR / "corpus.json")  # legacy snapshot, retained for rollback


def ensure_directories():
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


def _pdf_path(file_name, repository=None):
    if not file_name or Path(file_name).name != file_name or "/" in file_name or "\\" in file_name or not file_name.lower().endswith(".pdf"):
        raise ValueError("유효한 PDF 파일명이 필요합니다.")
    root = (repository or IndexRepository()).data_dir / "uploaded_pdfs"
    root.mkdir(parents=True, exist_ok=True)
    return root / file_name


def load_corpus_db():
    return IndexRepository().load().children


def get_index_status():
    snapshot = IndexRepository().load()
    return {"revision": snapshot.revision, **snapshot.manifest}


def is_file_already_uploaded(file_name):
    return _pdf_path(file_name).exists() or any(c["file_name"] == file_name for c in load_corpus_db())


def get_uploaded_papers_info():
    snapshot = IndexRepository().load()
    papers = {}
    for document in snapshot.documents:
        papers[document["file_name"]] = {"file_name": document["file_name"], "chunk_count": 0, "parent_count": 0, "parse_method": document["parse_method"], "size_kb": 0.0}
    for child in snapshot.children:
        record = papers.setdefault(child["file_name"], {"file_name": child["file_name"], "chunk_count": 0, "parent_count": 0, "parse_method": child.get("parse_method", "기본 파싱"), "size_kb": 0.0})
        record["chunk_count"] += 1
    for parent in snapshot.parents:
        papers[parent["file_name"]]["parent_count"] += 1
    for path in sorted((snapshot.path.parent.parent / "uploaded_pdfs" if not snapshot.legacy else snapshot.path / "uploaded_pdfs").glob("*.pdf")):
        record = papers.setdefault(path.name, {"file_name": path.name, "chunk_count": 0, "parent_count": 0, "parse_method": "미색인", "size_kb": 0.0})
        record["size_kb"] = path.stat().st_size / 1024
    return list(papers.values())


def _existing_documents(snapshot, repository, min_text_len_per_page):
    from src.parser import extract_document_from_pdf
    previous = {document["file_name"]: document for document in snapshot.documents}
    unchanged_pipeline = not snapshot.legacy and snapshot.manifest.get("pipeline_signature") == pipeline_signature()
    documents = {}
    for path in sorted((repository.data_dir / "uploaded_pdfs").glob("*.pdf")):
        content = path.read_bytes()
        cached = previous.get(path.name)
        if unchanged_pipeline and cached and cached["file_hash"] == hashlib.sha256(content).hexdigest():
            documents[path.name] = cached
        else:
            documents[path.name] = extract_document_from_pdf(content, path.name, min_text_len_per_page)
    return documents


def _replace_pdf(path, content):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        temporary.write_bytes(content)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_pdf(file_name, file_bytes, settings=None, min_text_len_per_page=50, overwrite=True, data_dir=None):
    repository = IndexRepository(data_dir) if data_dir is not None else IndexRepository()
    path = _pdf_path(file_name, repository)
    from src.parser import extract_document_from_pdf
    incoming = extract_document_from_pdf(file_bytes, file_name, min_text_len_per_page)
    with repository.writer():
        previous = repository.load()
        if path.exists() and not overwrite:
            raise FileExistsError("이미 등록된 논문입니다.")
        documents = _existing_documents(previous, repository, min_text_len_per_page)
        documents[file_name] = incoming
        settings = settings or ChunkSettings.from_dict(previous.manifest.get("settings"))
        from src.index_builder import build_snapshot
        snapshot = build_snapshot(list(documents.values()), settings, repository, previous=previous)
        original = path.read_bytes() if path.exists() else None
        try:
            _replace_pdf(path, file_bytes)
            repository.activate(snapshot)
        except Exception:
            if original is None:
                path.unlink(missing_ok=True)
            else:
                _replace_pdf(path, original)
            raise
    return [child for child in snapshot.children if child["file_name"] == file_name]


def remove_paper_from_db(file_name, data_dir=None):
    repository = IndexRepository(data_dir) if data_dir is not None else IndexRepository()
    path = _pdf_path(file_name, repository)
    with repository.writer():
        previous = repository.load()
        documents = _existing_documents(previous, repository, 50)
        documents.pop(file_name, None)
        from src.index_builder import build_snapshot
        snapshot = build_snapshot(list(documents.values()), ChunkSettings.from_dict(previous.manifest.get("settings")), repository, previous=previous)
        original = path.read_bytes() if path.exists() else None
        try:
            path.unlink(missing_ok=True)
            repository.activate(snapshot)
        except Exception:
            if original is not None:
                _replace_pdf(path, original)
            raise
