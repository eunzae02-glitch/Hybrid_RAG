"""Page-bound parent contexts and token-budgeted retrieval children."""

import re

from src.chunk_config import ChunkSettings
from src.document_structure import SEARCHABLE_KINDS
from src.token_budget import get_token_budget


def _make_parent(document, blocks, number, budget):
    pieces, spans, position = [], [], 0
    for block in blocks:
        if pieces:
            pieces.append("\n\n")
            position += 2
        text = block["text"]
        spans.append({"page_num": block["page_num"], "block_id": block["block_id"], "start": 0, "end": len(text), "bbox": block["bbox"], "parent_start": position, "parent_end": position + len(text)})
        pieces.append(text)
        position += len(text)
    text = "".join(pieces)
    first = blocks[0]
    return {
        "parent_id": f"{document['document_id']}-P{number:04d}", "document_id": document["document_id"],
        "file_name": document["file_name"], "page_num": first["page_num"],
        "section_id": first["section_id"], "section_title": first["section_title"],
        "kind": first["kind"], "text": text, "source_spans": spans,
        "parse_method": document["parse_method"], "token_count": budget.count(text, special=False), "child_ids": [],
    }


def build_parents(document, settings, budget):
    parents, pending = [], []

    def flush():
        if pending:
            parents.append(_make_parent(document, pending, len(parents) + 1, budget))
            pending.clear()

    for block in document["blocks"]:
        if block["kind"] not in SEARCHABLE_KINDS or not block["text"].strip():
            flush()
            continue
        if re.fullmatch(r"(?:Fig\.?|Figure|Table)\s*[\dIVX]+[.:]?", block["text"].strip(), re.IGNORECASE):
            flush()
            continue
        if pending:
            first = pending[0]
            same_group = all(block[key] == first[key] for key in ("page_num", "section_id", "kind"))
            combined = "\n\n".join([*(item["text"] for item in pending), block["text"]])
            if not same_group or budget.count(combined, special=False) > settings.parent_target_tokens:
                flush()
        pending.append(block)
    flush()
    return parents


def source_spans_for_child(parent, start, end):
    result = []
    for span in parent["source_spans"]:
        intersection_start, intersection_end = max(start, span["parent_start"]), min(end, span["parent_end"])
        if intersection_start >= intersection_end:
            continue
        result.append({
            "page_num": span["page_num"], "block_id": span["block_id"], "bbox": span["bbox"],
            "start": span["start"] + intersection_start - span["parent_start"],
            "end": span["start"] + intersection_end - span["parent_start"],
            "child_start": intersection_start - start, "child_end": intersection_end - start,
        })
    return result


def split_parent(parent, settings, budget):
    text = parent["text"]
    offsets = budget.offsets(text)
    if not offsets:
        return []
    prefix = budget.prefix(parent["section_title"], settings.section_prefix_tokens)
    sentence_ends = {match.start() for match in re.finditer(r"(?<=[.!?])(?:\s+|$)", text)}
    sentence_ends.add(len(text.rstrip()))
    children = []
    start_token, previous_end = 0, 0
    while start_token < len(offsets):
        start = 0 if start_token == 0 else offsets[start_token][0]
        end_token = min(len(offsets), start_token + settings.child_target_tokens)
        if settings.mode == "smart" and end_token < len(offsets):
            minimum_advance = max(settings.overlap_tokens + 1, settings.child_target_tokens // 2)
            boundaries = [i for i in range(start_token + minimum_advance, end_token + 1) if offsets[i - 1][1] in sentence_ends]
            if boundaries:
                end_token = boundaries[-1]
        while end_token > start_token:
            end = len(text) if end_token == len(offsets) else offsets[end_token][0]
            body = text[start:end]
            if budget.count(prefix + body) <= budget.limit:
                break
            end_token -= 1
        if end_token <= start_token:
            raise ValueError("1토큰도 담을 수 없는 임베딩 접두사 설정입니다.")
        child_id = f"{parent['parent_id']}-C{len(children) + 1:03d}"
        child = {
            "child_id": child_id, "chunk_id": child_id, "parent_id": parent["parent_id"],
            "document_id": parent["document_id"], "file_name": parent["file_name"],
            "section_id": parent["section_id"], "section_title": parent["section_title"],
            "page_num": parent["page_num"], "kind": parent["kind"], "searchable": True,
            "parse_method": parent["parse_method"], "text": body, "embedding_text": prefix + body,
            "token_count": budget.validate(prefix + body), "content_token_count": budget.count(body, special=False),
            "source_spans": source_spans_for_child(parent, start, end),
            "parent_start": start, "parent_end": end,
            "parent_token_start": start_token, "parent_token_end": end_token,
            "actual_overlap_tokens": max(0, previous_end - start_token) if children else 0,
            "sentence_split": end_token < len(offsets) and offsets[end_token - 1][1] not in sentence_ends,
        }
        children.append(child)
        if end_token == len(offsets):
            break
        overlap = min(settings.overlap_tokens, end_token - start_token - 1)
        previous_end = end_token
        start_token = end_token - overlap
    parent["child_ids"] = [child["child_id"] for child in children]
    return children


def build_document_chunks(document, settings=None, budget=None):
    settings = settings or ChunkSettings()
    budget = budget or get_token_budget()
    parents = build_parents(document, settings, budget)
    children = [child for parent in parents for child in split_parent(parent, settings, budget)]
    return parents, children
