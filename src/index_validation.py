"""Validate the complete evidence contract before an index is activated."""

from collections import Counter


def validate_documents(documents, parents, children, budget, settings=None):
    errors = []
    document_map = {doc["document_id"]: doc for doc in documents}
    block_map = {block["block_id"]: block for doc in documents for block in doc["blocks"]}
    parent_map = {parent["parent_id"]: parent for parent in parents}
    if len(parent_map) != len(parents) or len({c["child_id"] for c in children}) != len(children):
        errors.append("duplicate chunk identity")
    covered = {parent_id: bytearray(len(parent["text"])) for parent_id, parent in parent_map.items()}
    previous_children = {}
    maximum = 0
    for child in children:
        parent = parent_map.get(child["parent_id"])
        if parent is None:
            errors.append(f"missing parent: {child['child_id']}")
            continue
        length = budget.count(child["embedding_text"])
        maximum = max(maximum, length)
        if length > budget.limit or length != child["token_count"]:
            errors.append(f"token budget: {child['child_id']}")
        if child["page_num"] != parent["page_num"]:
            errors.append(f"cross-page child: {child['child_id']}")
        if child["document_id"] != parent["document_id"] or child["file_name"] != parent["file_name"]:
            errors.append(f"source document mismatch: {child['child_id']}")
        previous = previous_children.get(parent["parent_id"])
        overlap = max(0, previous["parent_token_end"] - child["parent_token_start"]) if previous else 0
        if overlap != child["actual_overlap_tokens"] or (settings and overlap > settings.overlap_tokens):
            errors.append(f"overlap mismatch: {child['child_id']}")
        previous_children[parent["parent_id"]] = child
        if parent["text"][child["parent_start"]:child["parent_end"]] != child["text"]:
            errors.append(f"parent offset mismatch: {child['child_id']}")
        if not child["source_spans"]:
            errors.append(f"missing provenance: {child['child_id']}")
        for span in child["source_spans"]:
            block = block_map.get(span["block_id"])
            if block is None or block["page_num"] != span["page_num"] or span["page_num"] != child["page_num"]:
                errors.append(f"source page mismatch: {child['child_id']}")
                continue
            if not 0 <= span["start"] < span["end"] <= len(block["text"]):
                errors.append(f"source range mismatch: {child['child_id']}")
            elif block["text"][span["start"]:span["end"]] != child["text"][span["child_start"]:span["child_end"]]:
                errors.append(f"source text mismatch: {child['child_id']}")
            if span["bbox"] != block["bbox"]:
                errors.append(f"source coordinates mismatch: {child['child_id']}")
        coverage = covered[parent["parent_id"]]
        coverage[child["parent_start"]:child["parent_end"]] = b"\x01" * (child["parent_end"] - child["parent_start"])
    for parent_id, parent in parent_map.items():
        document = document_map.get(parent["document_id"])
        if document is None or not 1 <= parent["page_num"] <= document["page_count"]:
            errors.append(f"invalid document page: {parent_id}")
        if any(not flag and not char.isspace() for flag, char in zip(covered[parent_id], parent["text"])):
            errors.append(f"uncovered parent text: {parent_id}")
    document_counts = Counter(child["document_id"] for child in children)
    for document in documents:
        if not document_counts[document["document_id"]]:
            errors.append(f"no searchable body extracted: {document['file_name']}")
    report = {"passed": not errors, "documents": len(documents), "parents": len(parents), "children": len(children), "max_input_tokens": maximum, "model_limit_tokens": budget.limit, "uncovered_parent_characters": 0 if not errors else None, "errors": errors[:30]}
    if errors:
        raise ValueError("색인 검증 실패: " + "; ".join(errors[:5]))
    return report
