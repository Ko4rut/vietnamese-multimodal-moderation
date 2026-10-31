"""Corpus-aware repeated web UI detection for Silver V2.

Document frequencies are collected from *distinct records*, never token counts.
Frequency alone never implies that a block is boilerplate: only a deliberately
small inventory of interface-like lines can become automatic deletion candidates.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_TEXT_SPACES = re.compile(r"\s+")
_UI_LABEL = re.compile(
    r"^(?:skip to content|đăng nhập|đăng ký|quên mật khẩu\??"
    r"|chính sách bảo mật|điều khoản (?:sử dụng|dịch vụ)"
    r"|bài viết liên quan|bài viết mới|xem nhiều|related posts"
    r"|trả lời hủy|email\s*\*|tên\s*\*|bình luận\s*\*"
    r"|tìm kiếm\s*:|giỏ hàng|chưa có sản phẩm trong giỏ hàng"
    r"|save my name, email\b.*|lưu tên của tôi, email\b.*"
    r"|email của bạn sẽ không được hiển thị công khai\b.*"
    r"|your email address will not be published\b.*"
    r"|copyright\s*(?:©|\(c\)|\d)|©\s*\d{4}\b.*"
    r"|powered by\b.*|quảng cáo|cookie policy|all rights reserved)"
    r"\s*$",
    re.IGNORECASE,
)
_BREADCRUMB = re.compile(
    r"^(?:trang chủ|home)\s*(?:[>/»｜|]|(?: / ))\s*",
    re.IGNORECASE,
)
_EXCLUDED = frozenset(
    {"tham khảo", "tài liệu tham khảo", "nguồn tham khảo",
     "references", "kết luận", "mục lục", "liên kết ngoài", "chú thích"}
)


def normalize_block(line: str) -> str:
    """Deterministic string key; do not alter original text in the output."""
    return _TEXT_SPACES.sub(" ", line.casefold()).strip()


def is_ui_candidate(line: str) -> bool:
    key = normalize_block(line)
    return (
        3 <= len(key) <= 180
        and key not in _EXCLUDED
        and bool(_UI_LABEL.match(key) or _BREADCRUMB.match(key))
    )


@dataclass(frozen=True, slots=True)
class CorpusV2Stats:
    documents: int
    min_document_frequency: int
    document_frequency: dict[str, int]
    head_frequency: dict[str, int]
    tail_frequency: dict[str, int]

    def is_repeated_ui(self, line: str, *, position: float) -> bool:
        key = normalize_block(line)
        if not is_ui_candidate(key):
            return False
        if self.document_frequency.get(key, 0) < self.min_document_frequency:
            return False
        # A repeated UI line inside an article may be genuine quoted content.
        if position <= 0.20:
            return self.head_frequency.get(key, 0) >= self.min_document_frequency
        if position >= 0.60:
            return self.tail_frequency.get(key, 0) >= self.min_document_frequency
        return False


def build_corpus_stats(
    records: Iterable[Mapping[str, Any]], *, min_document_frequency: int = 25,
) -> CorpusV2Stats:
    """Stream over the verified source once with bounded UI-only counters."""
    if min_document_frequency < 2:
        raise ValueError("min_document_frequency must be >= 2")
    counts: Counter[str] = Counter()
    head: Counter[str] = Counter()
    tail: Counter[str] = Counter()
    total = 0
    for record in records:
        text = record["text"]
        if not isinstance(text, str):
            raise ValueError("Corpus statistics require text strings")
        total += 1
        seen: set[str] = set()
        first: set[str] = set()
        last: set[str] = set()
        length = max(1, len(text))
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            key = normalize_block(raw_line)
            if is_ui_candidate(key):
                seen.add(key)
                fraction = pos / length
                if fraction <= 0.20:
                    first.add(key)
                if fraction >= 0.60:
                    last.add(key)
            pos += len(raw_line)
        counts.update(seen)
        head.update(first)
        tail.update(last)
    # Store only frequent candidates so report and cleaner memory stays bounded.
    frequent = {k: v for k, v in sorted(counts.items())
                if v >= min_document_frequency}
    return CorpusV2Stats(
        documents=total,
        min_document_frequency=min_document_frequency,
        document_frequency=frequent,
        head_frequency={k: head[k] for k in frequent},
        tail_frequency={k: tail[k] for k in frequent},
    )
