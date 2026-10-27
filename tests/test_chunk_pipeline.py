"""Contracts for token limits, source ranges, overlap and immutable publishing."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunk_config import ChunkSettings, build_fingerprint
from src.chunker_smart import build_document_chunks
from src.index_repository import FORMAT_VERSION, IndexIntegrityError, IndexRepository, IndexSnapshot, file_digest, write_json
from src.index_validation import validate_documents
from src.retrieval_quality import is_searchable_chunk
from src.token_budget import get_token_budget


def document(texts):
    blocks = []
    for index, (text, page, kind) in enumerate(texts):
        blocks.append({"block_id": f"block-{index}", "text": text, "page_num": page, "bbox": [10, 10, 500, 100], "section_id": "section", "section_title": "Results and discussion", "kind": kind, "parse_method": "test"})
    return {"document_id": "test-document", "file_name": "test.pdf", "file_hash": "source-hash", "page_count": max(page for _, page, _ in texts), "blocks": blocks, "parse_method": "test"}


def stub_revision(repository, children=None, passed=True):
    revision, path = repository.new_revision()
    values = {"documents": [], "parents": [], "children": children or []}
    for name, value in values.items():
        write_json(path / f"{name}.json", value)
    manifest = {"format_version": FORMAT_VERSION, "validation": {"passed": passed}, "files": {f"{name}.json": file_digest(path / f"{name}.json") for name in values}}
    write_json(path / "manifest.json", manifest)
    return IndexSnapshot(revision, path, manifest, **values)


class TokenChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.budget = get_token_budget()

    def test_full_input_and_long_sentences_fit_model(self):
        text = ("The gate voltage controls the transistor channel and the measured drain current " * 40) + "."
        doc = document([(text, 1, "body")])
        doc["blocks"][0]["section_title"] = "Very long section title " * 30
        parents, children = build_document_chunks(doc, ChunkSettings(), self.budget)
        report = validate_documents([doc], parents, children, self.budget)
        self.assertTrue(report["passed"])
        self.assertEqual(parents[0]["text"], text)
        self.assertGreater(len(children), 1)
        self.assertTrue(any(child["sentence_split"] for child in children))
        self.assertTrue(all(self.budget.count(child["embedding_text"]) <= 128 for child in children))

    def test_overlap_is_applied_in_both_modes(self):
        doc = document([("Threshold voltage and drain current are measured at each device temperature. " * 30, 1, "body")])
        for mode in ("smart", "fixed"):
            for overlap in (0, 16):
                parents, children = build_document_chunks(doc, ChunkSettings(overlap_tokens=overlap, mode=mode), self.budget)
                self.assertTrue(all(child["actual_overlap_tokens"] == overlap for child in children[1:]))
                for previous, current in zip(children, children[1:]):
                    self.assertEqual(previous["parent_token_end"] - current["parent_token_start"], overlap)
                validate_documents([doc], parents, children, self.budget)

    def test_short_blocks_never_merge_across_pages(self):
        doc = document([("Short first-page paragraph.", 1, "body"), ("Second-page paragraph with a measured drain current and a threshold voltage.", 2, "body")])
        parents, children = build_document_chunks(doc, budget=self.budget)
        self.assertEqual([parent["page_num"] for parent in parents], [1, 2])
        self.assertEqual({child["page_num"] for child in children}, {1, 2})
        validate_documents([doc], parents, children, self.budget)

    def test_short_numeric_caption_and_unicode_are_preserved(self):
        doc = document([("SS = 65 mV/dec at VDS = 0.1 V.", 1, "caption"), ("게이트 전압에 따른 누설 전류와 문턱 전압의 측정 결과를 비교합니다. " * 20, 1, "body")])
        parents, children = build_document_chunks(doc, budget=self.budget)
        self.assertTrue(any(child["text"] == doc["blocks"][0]["text"] and is_searchable_chunk(child) for child in children))
        validate_documents([doc], parents, children, self.budget)

    def test_source_mismatch_and_coverage_loss_are_rejected(self):
        doc = document([("The fabricated device has a measured threshold voltage and drain current. " * 20, 1, "body")])
        parents, children = build_document_chunks(doc, budget=self.budget)
        broken = deepcopy(children)
        broken[0]["source_spans"][0]["page_num"] = 2
        with self.assertRaisesRegex(ValueError, "source page mismatch"):
            validate_documents([doc], parents, broken, self.budget)
        with self.assertRaisesRegex(ValueError, "uncovered parent text"):
            validate_documents([doc], parents, children[1:], self.budget)

    def test_chunk_settings_change_the_build_fingerprint(self):
        docs = [document([("Device results.", 1, "body")])]
        baseline = build_fingerprint(docs, ChunkSettings(), "model-a")
        self.assertNotEqual(baseline, build_fingerprint(docs, ChunkSettings(overlap_tokens=0), "model-a"))
        self.assertNotEqual(baseline, build_fingerprint(docs, ChunkSettings(), "model-b"))

    def test_document_without_searchable_body_is_rejected_alongside_valid_document(self):
        good = document([("Measured transistor characteristics and drain current.", 1, "body")])
        empty = document([("Corresponding author.", 1, "author")])
        empty["document_id"], empty["file_name"] = "empty-document", "empty.pdf"
        empty["blocks"][0]["block_id"] = "empty-block"
        parents, children = build_document_chunks(good, budget=self.budget)
        with self.assertRaisesRegex(ValueError, "no searchable body extracted: empty.pdf"):
            validate_documents([good, empty], parents, children, self.budget)

    def test_incorrect_overlap_metadata_is_rejected(self):
        doc = document([("Threshold voltage and drain current are measured. " * 30, 1, "body")])
        parents, children = build_document_chunks(doc, budget=self.budget)
        children[1]["actual_overlap_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "overlap mismatch"):
            validate_documents([doc], parents, children, self.budget, ChunkSettings())


class PublishingTests(unittest.TestCase):
    def test_same_count_content_change_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            repository = IndexRepository(folder)
            snapshot = stub_revision(repository, [{"chunk_id": "a", "text": "before"}])
            repository.activate(snapshot)
            self.assertEqual(repository.load().children[0]["text"], "before")
            write_json(snapshot.path / "children.json", [{"chunk_id": "a", "text": "after"}])
            with self.assertRaises(IndexIntegrityError):
                repository.load()

    def test_failed_validation_does_not_replace_active_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            repository = IndexRepository(folder)
            good = stub_revision(repository)
            repository.activate(good)
            original = repository.pointer.read_bytes()
            bad = stub_revision(repository, passed=False)
            with self.assertRaises(IndexIntegrityError):
                repository.activate(bad)
            self.assertEqual(repository.pointer.read_bytes(), original)
            self.assertEqual(repository.load().revision, good.revision)

    def test_upload_restores_original_when_activation_fails(self):
        from src.storage import ingest_pdf
        with tempfile.TemporaryDirectory() as folder:
            repository = IndexRepository(folder)
            uploaded = Path(folder) / "uploaded_pdfs"
            uploaded.mkdir()
            original = uploaded / "test.pdf"
            original.write_bytes(b"original PDF bytes")
            incoming = document([("New document body.", 1, "body")])
            snapshot = stub_revision(repository)
            with patch("src.parser.extract_document_from_pdf", return_value=incoming), patch("src.storage._existing_documents", return_value={}), patch("src.index_builder.build_snapshot", return_value=snapshot), patch.object(IndexRepository, "activate", side_effect=OSError("activation failed")):
                with self.assertRaises(OSError):
                    ingest_pdf("test.pdf", b"replacement bytes", data_dir=folder)
            self.assertEqual(original.read_bytes(), b"original PDF bytes")
            self.assertFalse(repository.pointer.exists())


class RankingTests(unittest.TestCase):
    def test_duplicate_children_cannot_inflate_parent_score(self):
        from src.search_hybrid import group_hybrid_chunks
        items = [{"chunk": {"file_name": "test.pdf", "chunk_id": str(index), "parent_id": "same-parent"}, "rrf_score": score} for index, score in enumerate((0.02, 0.019, 0.018))]
        paper = group_hybrid_chunks(items, 3)[0]
        self.assertEqual(paper["unique_parent_count"], 1)
        self.assertEqual(paper["sum_rrf_score"], 0.02)

    def test_zero_weight_retriever_cannot_return_zero_score_papers(self):
        from src.search_hybrid import fuse_child_ranks
        vector = [{"chunk": {"file_name": "test.pdf", "chunk_id": "a"}, "similarity": 0.8}]
        self.assertEqual(fuse_child_ranks([], vector, 1.0, 0.0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
