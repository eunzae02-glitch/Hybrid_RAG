"""Shared, token-based configuration for indexing and the UI."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
PARSER_VERSION = "page-blocks-3"
CHUNKER_VERSION = "parent-child-2"


@dataclass(frozen=True)
class ChunkSettings:
    child_target_tokens: int = 100
    overlap_tokens: int = 16
    parent_target_tokens: int = 500
    mode: str = "smart"
    section_prefix_tokens: int = 12

    def __post_init__(self):
        if not 8 <= self.child_target_tokens <= 126:
            raise ValueError("검색 청크 목표 길이는 8~126토큰이어야 합니다.")
        if not 0 <= self.overlap_tokens < self.child_target_tokens:
            raise ValueError("겹침 토큰은 0 이상, 청크 목표 길이 미만이어야 합니다.")
        if self.parent_target_tokens < self.child_target_tokens:
            raise ValueError("원문 문맥의 목표 길이는 검색 청크보다 작을 수 없습니다.")
        if self.mode not in {"smart", "fixed"}:
            raise ValueError("지원하지 않는 청킹 모드입니다.")
        if not 0 <= self.section_prefix_tokens <= 24:
            raise ValueError("섹션 접두사는 최대 24토큰까지 허용됩니다.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value) if value else cls()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@lru_cache(maxsize=32)
def _file_hash(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model_dir=MODEL_DIR) -> str:
    folder = Path(model_dir)
    files = sorted([*folder.glob("*.json"), *folder.glob("*.safetensors"), *folder.glob("*.bin"), *folder.glob("*/*.json")])
    if not files:
        raise FileNotFoundError("내장 임베딩 모델을 찾을 수 없습니다.")
    return stable_hash({str(path.relative_to(folder)): _file_hash(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files})


def build_fingerprint(documents, settings, embedding_signature):
    return stable_hash({
        "sources": sorted((doc["file_name"], doc["file_hash"], doc.get("min_text_len_per_page", 50)) for doc in documents),
        "parser": PARSER_VERSION, "chunker": CHUNKER_VERSION,
        "settings": settings.to_dict(), "model": embedding_signature, "pipeline": pipeline_signature(),
    })


def pipeline_signature():
    names = ("parser.py", "document_structure.py", "chunker_smart.py", "token_budget.py", "index_validation.py", "vector_store.py")
    return stable_hash({name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in names})
