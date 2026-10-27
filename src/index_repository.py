"""Immutable index revisions selected by one atomically replaced manifest pointer."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FORMAT_VERSION = 2


class IndexIntegrityError(RuntimeError):
    pass


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_file(source, destination):
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class IndexSnapshot:
    revision: str
    path: Path
    manifest: dict
    documents: list
    parents: list
    children: list

    @property
    def legacy(self):
        return self.revision == "legacy"

    @property
    def child_map(self):
        return {child.get("child_id", str(child["chunk_id"])): child for child in self.children}

    @property
    def parent_map(self):
        return {parent["parent_id"]: parent for parent in self.parents}


class IndexRepository:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir).resolve()
        self.pointer = self.data_dir / "active_index.json"

    def revision_path(self, revision):
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[a-f0-9]{8}", revision):
            raise IndexIntegrityError("색인 버전 경로가 올바르지 않습니다.")
        return self.data_dir / "indexes" / revision

    def load(self):
        if not self.pointer.exists():
            legacy = self.data_dir / "corpus.json"
            children = json.loads(legacy.read_text(encoding="utf-8")) if legacy.exists() else []
            return IndexSnapshot("legacy", self.data_dir, {"format_version": 1}, [], [], children)
        pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
        return self.load_revision(pointer["revision"], pointer["manifest_sha256"])

    def load_revision(self, revision, expected_manifest_hash=None):
        path = self.revision_path(revision)
        if expected_manifest_hash is not None and file_digest(path / "manifest.json") != expected_manifest_hash:
            raise IndexIntegrityError("색인 manifest가 변경되었습니다. 이전 버전으로 복구하거나 재색인해 주세요.")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION or not manifest.get("validation", {}).get("passed"):
            raise IndexIntegrityError("검증을 통과하지 않은 색인입니다.")
        values = {}
        for name in ("documents", "parents", "children"):
            file = path / f"{name}.json"
            if file_digest(file) != manifest["files"][f"{name}.json"]:
                raise IndexIntegrityError(f"{name} 데이터가 변경되었습니다. 재색인해 주세요.")
            values[name] = json.loads(file.read_text(encoding="utf-8"))
        return IndexSnapshot(revision, path, manifest, **values)

    def new_revision(self):
        revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        path = self.revision_path(revision)
        path.mkdir(parents=True)
        return revision, path

    def activate(self, snapshot):
        manifest = json.loads((snapshot.path / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("validation", {}).get("passed"):
            raise IndexIntegrityError("검증되지 않은 색인은 활성화할 수 없습니다.")
        for name, expected in manifest["files"].items():
            if file_digest(snapshot.path / name) != expected:
                raise IndexIntegrityError("활성화 전 데이터 검증에 실패했습니다.")
        write_json(self.pointer, {"revision": snapshot.revision, "manifest_sha256": file_digest(snapshot.path / "manifest.json")})

    @contextmanager
    def writer(self, timeout=90):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / ".index-writer.lock").open("a+b") as lock:
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            started = time.monotonic()
            while True:
                try:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() - started >= timeout:
                        raise RuntimeError("다른 색인 작업이 진행 중입니다. 잠시 후 다시 시도해 주세요.")
                    time.sleep(0.1)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_UN)
