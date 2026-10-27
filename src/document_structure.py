import re

# 1. 측정 단위 및 수식 기호 (섹션 번호 및 헤더 제외 필터)
UNITS_EXCLUSION = {
    'nm', 'um', 'mm', 'cm', 'm', 'pm', 'angstrom',
    'ps', 'ns', 'us', 'ms', 's', 'min', 'hr',
    'v', 'kv', 'mv', 'a', 'ma', 'ua', 'na', 'pa',
    'w', 'mw', 'kw', 'hz', 'khz', 'mhz', 'ghz', 'thz',
    'k', 'c', 'db', 'bit', 'byte', 'kb', 'mb', 'gb', 'tb',
    'layer', 'layers', 'cell', 'cells', 'times', 'x',
    'k=60', 'k=30', 'wt%', 'ppm', 'deg', 'degree', 'degrees'
}

# 2. 로마자 섹션 번호 (I ~ XII)
ROMAN_NUMERALS = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii'}

# 3. 캡션 및 라틴 약어 제외 접두사
CAPTION_AND_LATIN_PREFIXES = (
    'fig.', 'figure', 'fig ', 'table', 'tab.', 'scheme', 'chart', 
    'graph', 'ref.', 'reference', 'i.e.', 'e.g.', 'etc.', 'et al.', 'cf.', 'vs.'
)

# 4. 학술 표준 주요 단독 키워드
STANDARD_SECTION_KEYWORDS = {
    'abstract', '초록', '개요', 'introduction', '서론', 
    'background', '배경', 'related work', 'related works',
    'device fabrication', 'process integration', 'experimental',
    'experimental results', 'results and discussion', 'results', 'discussion',
    'circuit simulation', 'simulation results', 'conclusion', 'conclusions', '결론'
}

PAGE_MARKER_REGEX = re.compile(r'---\s*\[Page\s+(\d+)(?:\s*\(.*?\))?\]\s*---', re.IGNORECASE)


def is_valid_section_header(line: str) -> tuple[bool, str]:
    """
    논문 섹션 헤더(제목)인지 정밀하게 판별합니다.
    
    1. 측정 단위(522 nm for k=60 등), 캡션(Fig 1), 수식, 라틴 약어(i.e., e.g.) 오탐 원천 차단
    2. 로마자(III. PROCESS INTEGRATION), 아라비아 숫자(1. Introduction, 2.1 Device Structure) 완벽 인식
    3. 표준 학술 헤더(Abstract, Conclusion 등) 인식
    
    Returns:
        (is_header: bool, clean_title: str)
    """
    raw = line.strip()
    if not raw or len(raw) > 100:
        return False, ""
        
    lower = raw.lower()
    
    # 캡션(Fig, Table), 라틴 약어(i.e., e.g.), 괄호, 참조문헌 번호, 페이지 번호 제외
    if lower.startswith(CAPTION_AND_LATIN_PREFIXES):
        return False, ""
    if raw.startswith('[') or raw.startswith('(') or raw.startswith('{'):
        return False, ""
    if re.match(r'^-?\s*\d+\s*/\s*\d+\s*-?$', raw):  # - 1 / 10 -
        return False, ""
    if re.match(r'^(?:page\s+)?\d+$', lower):
        return False, ""
        
    # 벤치마크 헤더 (BENCHMARK [A-BM25 Trap]: ...)
    if raw.startswith("BENCHMARK"):
        return True, raw
        
    # 로마자 섹션 헤더 (예: I. INTRODUCTION, III. PROCESS INTEGRATION, IV. CIRCUIT SIMULATION)
    roman_match = re.match(r'^([IVXLCDM]+)[\.\s:]+\s*([A-Za-z가-힣].*)$', raw)
    if roman_match:
        rom_num = roman_match.group(1).lower()
        title_text = roman_match.group(2).strip()
        if rom_num in ROMAN_NUMERALS and len(title_text) >= 3:
            first_w = title_text.split()[0].lower()
            if first_w not in UNITS_EXCLUSION:
                if not (raw.endswith('.') and len(raw.split()) > 6):
                    return True, raw
                
    # Section 명시적 키워드 (예: Section 1.1: Physical Transport, Section 2:)
    sec_word_match = re.match(r'^Section\s+(\d+(?:\.\d+)*|[IVXLCDM]+)[\.\s:]*\s*(.*)$', raw, re.IGNORECASE)
    if sec_word_match:
        return True, raw
        
    # 아라비아 숫자 섹션 헤더 (예: 1. Introduction, 2.1 Device Structure)
    # 단, 섹션 번호는 1~30 이하이어야 하며, 단위(nm, V 등)가 아니어야 함
    num_match = re.match(r'^(\d+(?:\.\d+){0,3})[\.\s:]+\s*([A-Za-z가-힣].*)$', raw)
    if num_match:
        sec_num_str = num_match.group(1)
        title_text = num_match.group(2).strip()
        main_num = int(sec_num_str.split('.')[0])
        if 1 <= main_num <= 30 and len(title_text) >= 3:
            first_w = title_text.split()[0].lower().rstrip(',.;:')
            if first_w not in UNITS_EXCLUSION and not re.match(r'^\d', first_w):
                if not (raw.endswith('.') and len(raw.split()) > 7) and (title_text[0].isupper() or ord(title_text[0]) >= 0xAC00):
                    return True, raw

    # 학술 표준 단독 키워드 (Abstract, Introduction, Conclusion 등)
    if lower.rstrip(':') in STANDARD_SECTION_KEYWORDS:
        return True, raw
        
    return False, ""


SEARCHABLE_KINDS = {"body", "caption", "table"}
REFERENCE_HEADING = re.compile(r"^(?:(?:\d+|[IVXLCDM]+)[.\s:]*)?(?:references?|bibliography|literature cited|참\s*고\s*문\s*헌)\s*[:.]?$", re.IGNORECASE)
CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:e[ -]?mail|corresponding author|department of|is with the|are with the|phone|manuscript received|date of publication|digital object identifier|work was supported|all rights reserved)\b", re.IGNORECASE)
RESEARCH = re.compile(r"\b(?:abstract|we (?:propose|report|present|demonstrate|investigate)|measured|fabricated|simulated|evaluated|observed|threshold voltage|gate oxide|leakage current|carrier mobility)\b", re.IGNORECASE)


def classify_block(text):
    stripped = text.strip()
    first = stripped.splitlines()[0] if stripped else ""
    if REFERENCE_HEADING.fullmatch(first):
        return "reference_heading"
    if re.fullmatch(r"[IVXLCDM]+\.", stripped):
        return "heading_marker"
    if CONTACT.search(stripped) and not RESEARCH.search(stripped):
        return "author"
    if re.match(r"^(?:table|tab\.)\s+[\dIVX]", first, re.IGNORECASE):
        return "table"
    if re.match(r"^(?:fig\.?|figure)\s*\d", first, re.IGNORECASE):
        return "caption"
    return "body"


