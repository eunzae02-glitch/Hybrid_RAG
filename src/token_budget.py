"""Measure the exact embedding input, including prefix and special tokens."""

from functools import lru_cache
import json
from pathlib import Path

from src.chunk_config import MODEL_DIR


class TokenBudget:
    def __init__(self, tokenizer=None, limit=None, model_dir=MODEL_DIR):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.tokenizer = tokenizer
        configured = int(getattr(tokenizer, "model_max_length", 128))
        if configured > 100_000:
            configured = 128
        config_file = Path(model_dir) / "sentence_bert_config.json"
        if config_file.exists():
            configured = min(configured, json.loads(config_file.read_text(encoding="utf-8")).get("max_seq_length", configured))
        self.limit = min(configured, limit) if limit is not None else configured

    def count(self, text, special=True):
        return len(self.tokenizer.encode(text, add_special_tokens=special, truncation=False, verbose=False))

    def offsets(self, text):
        result = self.tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
        return [(int(start), int(end)) for start, end in result["offset_mapping"] if end > start]

    def prefix(self, section_title, maximum_tokens):
        if maximum_tokens <= 0:
            return ""
        title = " ".join(section_title.split())
        offsets = self.offsets(title)
        if offsets:
            title = title[:offsets[min(len(offsets), max(1, maximum_tokens - 2)) - 1][1]]
        prefix = f"[{title}]\n" if title else ""
        while prefix and self.count(prefix, special=False) > maximum_tokens:
            title = title[:-1].rstrip()
            prefix = f"[{title}]\n" if title else ""
        return prefix

    def validate(self, text):
        length = self.count(text)
        if length > self.limit:
            raise ValueError(f"임베딩 입력이 {length}토큰으로 모델 한도 {self.limit}토큰을 초과했습니다.")
        return length


@lru_cache(maxsize=1)
def get_token_budget():
    return TokenBudget()
