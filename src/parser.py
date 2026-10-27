"""Extract ordered, page-bound text blocks with source coordinates."""

from collections import Counter
from functools import lru_cache
import hashlib
import io
import re
import statistics

import pymupdf as fitz

from src.chunk_config import PARSER_VERSION
from src.document_structure import REFERENCE_HEADING, classify_block, is_valid_section_header


class DocumentParseError(ValueError):
    pass


@lru_cache(maxsize=1)
def get_rapid_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as error:
        raise DocumentParseError("RapidOCR를 불러올 수 없습니다. 내장 실행 환경을 확인해 주세요.") from error
    try:
        return RapidOCR()
    except Exception as error:
        raise DocumentParseError("RapidOCR를 초기화하지 못했습니다. 내장 OCR 모델과 실행 환경을 확인해 주세요.") from error


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _text_blocks(page):
    result = []
    for raw in page.get_text("dict", sort=False)["blocks"]:
        if raw.get("type") != 0:
            continue
        lines = []
        for line in raw.get("lines", []):
            text = re.sub(r"[ \t]+", " ", "".join(span.get("text", "") for span in line.get("spans", []))).strip()
            if text:
                lines.append({"text": text, "bbox": list(line["bbox"]), "font_size": max((span.get("size", 10) for span in line.get("spans", [])), default=10)})
        if lines:
            result.append({"text": "\n".join(line["text"] for line in lines), "bbox": _union([line["bbox"] for line in lines]), "lines": lines})
    return result


def _ocr_blocks(page):
    from PIL import Image
    import numpy as np

    pixmap = page.get_pixmap(dpi=180)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    x_scale, y_scale = page.rect.width / image.width, page.rect.height / image.height
    engine = get_rapid_ocr_engine()
    try:
        result, _ = engine(np.array(image))
    except Exception as error:
        raise DocumentParseError("RapidOCR로 스캔 페이지를 읽지 못했습니다.") from error
    detections = []
    for points, text, _confidence in result or []:
        if str(text).strip():
            detections.append((str(text).strip(), [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]))
    blocks = []
    for text, box in detections:
        bbox = [float(box[0] * x_scale), float(box[1] * y_scale), float(box[2] * x_scale), float(box[3] * y_scale)]
        line = {"text": text, "bbox": bbox, "font_size": max(5.0, bbox[3] - bbox[1])}
        blocks.append({"text": text, "bbox": bbox, "lines": [line]})
    if not blocks:
        raise DocumentParseError("스캔 페이지에서 텍스트를 추출하지 못했습니다.")
    return blocks, "RapidOCR"


def order_page_blocks(blocks, page_width):
    """Read each detected column top to bottom; spanning blocks delimit bands."""
    middle = page_width / 2
    narrow = [b for b in blocks if b["bbox"][2] - b["bbox"][0] < page_width * 0.64 and len(b["text"].split()) >= 5]
    left = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 < middle]
    right = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 >= middle]
    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])), "single_column"
    left_extent = (min(b["bbox"][1] for b in left), max(b["bbox"][3] for b in left))
    right_extent = (min(b["bbox"][1] for b in right), max(b["bbox"][3] for b in right))
    if min(left_extent[1], right_extent[1]) <= max(left_extent[0], right_extent[0]):
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])), "single_column"
    spanning = [b for b in blocks if b["bbox"][0] < middle < b["bbox"][2] and b["bbox"][2] - b["bbox"][0] >= page_width * 0.64]
    remaining = [b for b in blocks if b not in spanning]

    def columns(items):
        return sorted(items, key=lambda b: ((b["bbox"][0] + b["bbox"][2]) / 2 >= middle, b["bbox"][1], b["bbox"][0]))

    ordered = []
    for anchor in sorted(spanning, key=lambda b: b["bbox"][1]):
        above = [b for b in remaining if b["bbox"][3] <= anchor["bbox"][1] + 2]
        ordered.extend(columns(above))
        remaining = [b for b in remaining if b not in above]
        ordered.append(anchor)
    ordered.extend(columns(remaining))
    return ordered, "two_column_heuristic"


def _split_structural_blocks(blocks, median_font):
    result = []
    for block in blocks:
        block_kind = classify_block(block["text"])
        pending = []
        def flush():
            if pending:
                result.append({"text": "\n".join(line["text"] for line in pending), "bbox": _union([line["bbox"] for line in pending]), "kind": block_kind})
                pending.clear()
        for line in block["lines"]:
            heading, _ = is_valid_section_header(line["text"])
            reference = bool(REFERENCE_HEADING.fullmatch(line["text"].strip()))
            large_title = line["font_size"] > median_font * 1.35 and len(line["text"]) < 140 and block_kind == "body"
            if block_kind != "author" and (heading or reference or large_title):
                flush()
                result.append({"text": line["text"], "bbox": line["bbox"], "kind": "reference_heading" if reference else "heading"})
            else:
                pending.append(line)
        flush()
    return result


def extract_document_from_pdf(pdf_file, file_name="document.pdf", min_text_len_per_page=50):
    if isinstance(pdf_file, bytes):
        content = pdf_file
    elif hasattr(pdf_file, "read"):
        content = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
    else:
        raise DocumentParseError("PDF 바이트 또는 파일 객체가 필요합니다.")
    file_hash = hashlib.sha256(content).hexdigest()
    document_id = hashlib.sha256((file_name + ":" + file_hash).encode("utf-8")).hexdigest()[:16]
    pages, all_blocks, methods = [], [], set()
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentParseError("암호화된 PDF는 먼저 암호를 해제해 주세요.")
            for page_index, page in enumerate(document):
                blocks = _text_blocks(page)
                extracted_length = sum(len(re.sub(r"\s", "", b["text"])) for b in blocks)
                method = "기본 파싱"
                if extracted_length < min_text_len_per_page and page.get_images():
                    blocks, method = _ocr_blocks(page)
                ordered, layout = order_page_blocks(blocks, page.rect.width)
                fonts = [line["font_size"] for b in ordered for line in b["lines"] for _ in range(max(1, min(len(line["text"]), 100)))]
                structured = _split_structural_blocks(ordered, statistics.median(fonts) if fonts else 10)
                for index, block in enumerate(structured):
                    block.update({"block_id": f"{document_id}:p{page_index + 1}:b{index + 1}", "page_num": page_index + 1, "parse_method": method})
                    all_blocks.append(block)
                pages.append({"page_num": page_index + 1, "width": page.rect.width, "height": page.rect.height, "layout": layout, "parse_method": method})
                methods.add(method)
    except DocumentParseError:
        raise
    except Exception as error:
        raise DocumentParseError(f"PDF를 읽지 못했습니다: {type(error).__name__}") from error

    repeating = Counter()
    for block in all_blocks:
        page = pages[block["page_num"] - 1]
        if block["bbox"][1] < page["height"] * 0.08 or block["bbox"][3] > page["height"] * 0.94:
            repeating[re.sub(r"\d+", "#", block["text"].lower())] += 1
    total_chars = sum(len(b["text"]) for b in all_blocks)
    seen_chars, section_number = 0, 0
    section_title = "본문"
    in_references = False
    for block in all_blocks:
        normalized = re.sub(r"\d+", "#", block["text"].lower())
        if repeating[normalized] >= max(2, len(pages) * 0.5) and len(block["text"]) < 250:
            block["kind"] = "header_footer"
        if block["kind"] == "reference_heading" and seen_chars >= total_chars * 0.4:
            in_references = True
        if in_references:
            block["kind"] = "reference"
        elif block["kind"] == "heading":
            section_number += 1
            section_title = block["text"]
        block["section_id"] = f"{document_id}:s{section_number}"
        block["section_title"] = section_title
        seen_chars += len(block["text"])
    if not any(b["text"].strip() for b in all_blocks):
        raise DocumentParseError("PDF에서 읽을 수 있는 본문을 찾지 못했습니다.")
    return {"document_id": document_id, "file_name": file_name, "file_hash": file_hash, "page_count": len(pages), "pages": pages, "blocks": all_blocks, "parse_method": " / ".join(sorted(methods)), "parser_version": PARSER_VERSION, "min_text_len_per_page": min_text_len_per_page}


def extract_text_from_pdf(pdf_file, min_text_len_per_page=50):
    """Compatibility view; indexing consumes extract_document_from_pdf instead."""
    document = extract_document_from_pdf(pdf_file, min_text_len_per_page=min_text_len_per_page)
    text = "\n\n".join(f"--- [Page {block['page_num']}] ---\n{block['text']}" for block in document["blocks"] if block["kind"] not in {"reference", "header_footer"})
    return text, document["parse_method"]
