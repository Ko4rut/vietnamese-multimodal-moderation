"""Conservative, auditable Vietnamese web-boilerplate tail cleaner.

Only standalone structural headings or unmistakable footer lines qualify.
A weak cue is review-only; never delete content from the middle of a story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .model_v2 import SilverV2CleaningDecision

RULESET_VERSION = "silver-v2-tail-v1"

# Keep these anchors strict and line-oriented. A mention of "bản quyền"
# inside an article, for example, must never trigger removal.
_STRONG = {
    "copyright": re.compile(
        r"^(?:©\s*\d{4}\b|copyright\s*(?:©|\(c\)|\d{4})"
        r"|bản quyền\s*(?:©|thuộc về|:)|all rights reserved\b"
        r"|powered by\b|được phát triển bởi\b|được cung cấp bởi\b)",
        re.IGNORECASE,
    ),
    "comment_form": re.compile(
        r"^(?:leave a reply\b|cancel reply\b|post comment\b"
        r"|save my name,\s*email\b|email của bạn sẽ không được hiển thị"
        r"|lưu tên của tôi,\s*email\b|mời bạn viết bình luận\b)",
        re.IGNORECASE,
    ),
}
_WEAK = {
    "related_articles": re.compile(
        r"^(?:bài viết (?:liên quan|mới|khác)|tin (?:liên quan|mới nhất)"
        r"|các tin khác|xem nhiều|post navigation|check also|you may also like"
        r"|có thể bạn quan tâm)\s*[:：-]?\s*$",
        re.IGNORECASE,
    ),
    "navigation": re.compile(
        r"^(?:trang chủ\s*\|\s*|chính sách bảo mật\s*(?:\||$)"
        r"|điều khoản sử dụng\s*(?:\||$)|liên hệ quảng cáo\s*(?:\||$)"
        r"|quảng cáo\s*$)",
        re.IGNORECASE,
    ),
    "html_js": re.compile(
        r"^(?:<script\b|</script>|<footer\b|</footer>|<form\b)",
        re.IGNORECASE,
    ),
}
_ALL = {**_STRONG, **_WEAK}


@dataclass(frozen=True, slots=True)
class SilverV2CleanerConfig:
    min_retained_chars: int = 200
    max_removed_fraction: float = 0.35
    tail_start_fraction: float = 0.55

    def __post_init__(self) -> None:
        if self.min_retained_chars < 1:
            raise ValueError("min_retained_chars must be positive")
        if not 0 < self.max_removed_fraction < 1:
            raise ValueError("max_removed_fraction must be between 0 and 1")
        if not 0 < self.tail_start_fraction < 1:
            raise ValueError("tail_start_fraction must be between 0 and 1")


class SilverV2Cleaner:
    """Only trim a guarded suffix; otherwise preserve text and request human review."""

    def __init__(self, config: SilverV2CleanerConfig | None = None) -> None:
        self.config = config or SilverV2CleanerConfig()

    def clean(self, text: str) -> SilverV2CleaningDecision:
        if not text.strip():
            return SilverV2CleaningDecision(text=text, action="unchanged", signals=(),
                                            removed_chars=0, removed_preview="")
        lines = text.splitlines(keepends=True)
        positions: list[int] = []
        pos = 0
        for line in lines:
            positions.append(pos)
            pos += len(line)

        matches: list[tuple[int, int, str, Literal["strong", "weak"]]] = []
        for index, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped:
                continue
            for key, expression in _ALL.items():
                if expression.match(stripped):
                    kind: Literal["strong", "weak"] = (
                        "strong" if key in _STRONG else "weak"
                    )
                    matches.append((positions[index], index, key, kind))
                    break

        # A footer must be at the tail. A cue at the beginning / inside
        # an article is deliberately ignored, not used as a cutting position.
        tail = [m for m in matches if m[0] >= len(text) * self.config.tail_start_fraction]
        if not tail:
            return SilverV2CleaningDecision(text=text, action="unchanged", signals=(),
                                            removed_chars=0, removed_preview="")

        signals = tuple(sorted({m[2] for m in tail}))
        strong = [m for m in tail if m[3] == "strong"]
        if not strong:
            return SilverV2CleaningDecision(text=text, action="review", signals=signals,
                                            removed_chars=0, removed_preview="")

        # Do not cut at a weak heading alone. An earlier weak marker is safe
        # only if another strong footer indicator follows it in the same tail.
        first_strong = strong[0]
        candidates = [
            m for m in tail if m[0] <= first_strong[0] and m[2] == "related_articles"
        ]
        start = candidates[0][0] if candidates else first_strong[0]
        kept = text[:start].rstrip()
        removed = len(text) - len(kept)
        if (
            len(kept) < self.config.min_retained_chars
            or removed > self.config.max_removed_fraction * len(text)
        ):
            return SilverV2CleaningDecision(text=text, action="guardrail", signals=signals,
                                            removed_chars=0, removed_preview="")

        return SilverV2CleaningDecision(
            text=kept,
            action="cleaned",
            signals=signals,
            removed_chars=removed,
            removed_preview=text[len(kept):].strip()[:240],
        )
