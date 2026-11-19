"""LLM-assisted News refinement for V3."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import yaml

V3_CONFIG = Path(__file__).resolve().parents[1] / "config" / "domains_v3.yaml"


def load_v3_labels() -> tuple[str, ...]:
    config = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    labels = config["domains"]
    if len(labels) != config["num_classes"] or len(set(labels)) != len(labels):
        raise ValueError("Invalid domains_v3.yaml")
    return tuple(labels)


NEWS_LABELS = (
    "News_Politics",
    "News_Crime_Disaster",
    "News_Current_Affairs",
)

_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_STOPWORDS = frozenset(
    "những các một của và là có cho với trong được theo từ về này đó khi đã "
    "sẽ vào tại người bài tin ngày tháng năm sau trước đang như nhưng còn "
    "không thì rằng hơn rất nhiều".split()
)


def sample_news_records(
    records: Iterable[Mapping[str, object]],
    *,
    sample_size: int = 300,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return a deterministic reservoir sample for taxonomy/guideline piloting."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    sample: list[dict[str, object]] = []
    seen = 0
    for row in records:
        if row.get("domain") != "News":
            continue
        seen += 1
        item = dict(row)
        if len(sample) < sample_size:
            sample.append(item)
            continue
        replace_at = rng.randrange(seen)
        if replace_at < sample_size:
            sample[replace_at] = item
    return sorted(sample, key=lambda row: str(row.get("id", "")))


def build_keyword_report(
    records: Iterable[Mapping[str, object]],
    *,
    top_n: int = 30,
    examples_per_domain: int = 3,
) -> dict[str, object]:
    """Summarize class-specific terms and examples; never assign or change labels."""
    if top_n < 1 or examples_per_domain < 0:
        raise ValueError("top_n must be positive and examples_per_domain nonnegative")

    documents: dict[str, list[dict[str, object]]] = defaultdict(list)
    class_terms: dict[str, Counter[str]] = defaultdict(Counter)
    class_document_counts: Counter[str] = Counter()
    global_terms: Counter[str] = Counter()
    total_documents = 0
    for row in records:
        domain = row.get("domain")
        text = row.get("text")
        if not isinstance(domain, str) or not domain:
            raise ValueError("Every record must have a non-empty domain")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every record must have non-empty text")
        tokens = [token.casefold() for token in _TOKEN.findall(text)]
        terms = {
            token for token in tokens
            if token not in _STOPWORDS and len(token) > 2
        }
        terms.update(
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
            if left not in _STOPWORDS and right not in _STOPWORDS
        )
        class_terms[domain].update(terms)
        class_document_counts[domain] += 1
        global_terms.update(terms)
        total_documents += 1
        if len(documents[domain]) < examples_per_domain:
            documents[domain].append({
                "id": str(row.get("id", "")),
                "text_excerpt": text[:500],
            })

    result: dict[str, object] = {
        "method": "unicode_token_document_frequency_and_class_specificity",
        "keyword_based_label_assignment": False,
        "total_documents": total_documents,
        "classes": {},
        "news_vs_other_class_keyword_overlap": {},
        "limitations": (
            "Keyword frequency and short examples are discovery signals only. "
            "They do not determine the correct label; review full articles and "
            "apply an approved labeling guide."
        ),
    }

    for domain, terms in sorted(class_terms.items()):
        class_size = max(1, class_document_counts[domain])
        ranked = sorted(
            terms.items(),
            key=lambda item: (
                (item[1] / class_size) / (global_terms[item[0]] / total_documents),
                item[1],
                item[0],
            ),
            reverse=True,
        )[:top_n]
        result["classes"][domain] = {
            "documents": class_size,
            "keywords": [
                {
                    "term": term,
                    "document_frequency": frequency,
                    "within_class_rate": frequency / class_size,
                    "global_document_frequency": global_terms[term],
                    "specificity_ratio": (
                        (frequency / class_size)
                        / (global_terms[term] / total_documents)
                    ),
                }
                for term, frequency in ranked
            ],
            "examples": documents[domain],
        }

    news_terms = class_terms.get("News", Counter())
    overlaps = {}
    for domain, frequencies in class_terms.items():
        if domain == "News":
            continue
        shared = sorted(
            set(news_terms) & set(frequencies),
            key=lambda term: (news_terms[term] + frequencies[term], term),
            reverse=True,
        )[:top_n]
        if shared:
            union = set(news_terms) | set(frequencies)
            overlaps[domain] = {
                "shared_top_terms": shared,
                "shared_term_jaccard": len(set(news_terms) & set(frequencies)) / len(union),
                "news_term_vocabulary": len(news_terms),
                "other_term_vocabulary": len(frequencies),
            }
    result["news_vs_other_class_keyword_overlap"] = dict(
        sorted(
            overlaps.items(),
            key=lambda item: item[1]["shared_term_jaccard"],
            reverse=True,
        )
    )
    return result


def build_prompt(
    record: Mapping[str, object],
    labels: Iterable[str] | None = None,
    *,
    max_chars: int = 12_000,
) -> str:
    """Build a strict, machine-readable labeling prompt for one record."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    allowed = list(labels or load_v3_labels())
    text = str(record.get("text", "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[Đã lược bớt phần còn lại của bài viết.]"
    return (
        "Bạn là người gán nhãn chủ đề cho bài viết tiếng Việt. "
        "Chỉ chọn đúng một nhãn trong danh sách sau. Không tự tạo nhãn mới.\n"
        f"Danh sách: {json.dumps(allowed, ensure_ascii=False)}\n\n"
        "Quy tắc:\n"
        "- Sports: dùng cho bài có chủ đề chính là thể thao.\n"
        "- News_Politics: bầu cử, đảng phái, tranh luận/chuyển biến chính trị hoặc ngoại giao.\n"
        "- News_Crime_Disaster: tội phạm, tai nạn, thiên tai hoặc tình huống khẩn cấp.\n"
        "- News_Current_Affairs: sự kiện thời sự không thuộc nhãn chuyên ngành cũ hoặc hai nhãn News trên.\n"
        "- Ưu tiên nhãn chuyên ngành cũ như Finance, Health, Business_and_Industrial, Sports "
        "khi chủ đề chính phù hợp; Law_and_Government dành cho luật, chính sách/quản trị và dịch vụ công.\n"
        "- Không suy đoán ngoài nội dung; nếu không chắc, đặt needs_review=true.\n\n"
        'Trả về JSON duy nhất: {"label":"...",'
        '"confidence":0.0,"needs_review":true,"reason":"..."}\n\n'
        f"Nội dung:\n{text[:5000]}"
    )


def validate_decisions(rows: Iterable[Mapping[str, object]], labels: Iterable[str] | None = None) -> list[dict]:
    """Validate and normalize LLM decisions without accepting unknown labels."""
    allowed = set(labels or load_v3_labels())
    validated: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("id", "")).strip()
        label = row.get("label")
        confidence = row.get("confidence")
        if not record_id:
            raise ValueError("Decision is missing id")
        if record_id in seen:
            raise ValueError(f"Duplicate decision id: {record_id}")
        if label not in allowed:
            raise ValueError(f"Unknown V3 label for {record_id}: {label}")
        if type(confidence) not in (int, float) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid confidence for {record_id}")
        seen.add(record_id)
        validated.append({
            "id": record_id,
            "label": str(label),
            "confidence": float(confidence),
            "needs_review": bool(row.get("needs_review", False)),
            "approved": row.get("approved") is True,
            "reviewer": str(row.get("reviewer", ""))[:200],
            "reason": str(row.get("reason", ""))[:500],
        })
    return validated
