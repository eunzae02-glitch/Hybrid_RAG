"""Exclude document scaffolding from evidence without changing stored papers."""

import re

_SECTION_PREFIX = re.compile(r"^\[[^\n]+\]\s*\n")
_BOILERPLATE = re.compile(
    r"\b(?:e[ -]?mail|phone|corresponding author|department of|is with the|are with the|"
    r"date of publication|date of current version|manuscript received|"
    r"review of this|work was supported|digital object identifier)\b|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE,
)
_RESEARCH_CONTENT = re.compile(
    r"\b(?:abstract|we (?:propose|report|demonstrate|investigate|present|show)|"
    r"measured|fabricated|simulated|evaluated|observed|"
    r"results? (?:show|indicate|demonstrate)|threshold voltage|gate oxide|"
    r"leakage current|carrier mobility|experimental results)\b|"
    r"(?:측정한|측정 결과|제안한다|실험 결과|문턱 전압|누설 전류)", re.IGNORECASE,
)


def is_research_content(text: str) -> bool:
    body = _SECTION_PREFIX.sub("", text.strip()).strip()
    if len(body) < 80 or len(re.findall(r"\w+", body)) < 12:
        return False
    if "참고문헌(References) 섹션 자동 절사 완료" in body and len(body) < 150:
        return False
    if _BOILERPLATE.search(body) and not _RESEARCH_CONTENT.search(body):
        return False
    return True


def is_searchable_chunk(chunk: dict) -> bool:
    if "parent_id" in chunk and "searchable" in chunk:
        return bool(chunk["searchable"]) and chunk.get("kind") in {"body", "caption", "table"}
    return is_research_content(chunk.get("text", ""))


def normalize_search_query(query: str) -> str:
    """Remove a conversational suffix while keeping the technical question."""
    cleaned = re.sub(
        r"\s*(?:에\s*대해(?:서)?\s*)?(?:알려\s*줘(?:요)?|알려\s*주세요|"
        r"설명해\s*줘(?:요)?|설명해\s*주세요|설명해\s*줄래(?:요)?)[.!?\s]*$",
        "", query.strip(),
    ).strip()
    return cleaned or query.strip()
