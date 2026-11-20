"""Synthetic Vietnamese Data Lake for the Bronze News EDA tests (no real article text)."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

TOPICS = {
    "politics": "quốc hội đại biểu bầu cử cử tri chính phủ thủ tướng chủ tịch đảng nghị sĩ ngoại giao "
                "đại sứ hội nghị thượng đỉnh hiệp định lãnh đạo bộ trưởng biểu quyết nghị viện".split(),
    "accident": "tai nạn cháy nổ nạn nhân tử vong thương vong cứu hộ cứu nạn hiện trường sơ tán "
                "bão lũ sạt lở ngập động đất thiệt hại giao thông xe tải đường cao tốc".split(),
    "sports": "bóng đá cầu thủ huấn luyện đội tuyển trận đấu giải đấu vô địch bàn thắng thủ môn "
              "trọng tài mùa giải huy chương ghi bàn sân vận động chung kết hiệp đấu".split(),
    "economy": "doanh nghiệp công ty sản xuất thị trường xuất khẩu đầu tư nhà máy ngân hàng lãi suất "
               "chứng khoán cổ phiếu tăng trưởng lợi nhuận khách hàng sản phẩm thương hiệu".split(),
    "law": "pháp luật nghị định thông tư quy định văn bản ban hành thủ tục hành chính quyết định "
           "chính sách thanh tra xử phạt vi phạm tòa án xét xử dự thảo hiệu lực sửa đổi".split(),
    "society": "xã hội gia đình trẻ em phụ nữ cộng đồng văn hóa truyền thống thanh niên học sinh "
               "từ thiện hôn nhân dân tộc đời sống nhân dân tình nguyện".split(),
    "sensitive": "bạo lực xâm hại khủng bố chiến tranh vũ khí nghiện mại dâm cực đoan quấy rối "
                 "tố cáo nạn nhân thương tích tự tử".split(),
    "health": "bệnh viện bác sĩ bệnh nhân điều trị vắc xin dịch bệnh thuốc y tế triệu chứng "
              "phẫu thuật dinh dưỡng sức khỏe".split(),
}
FILLER = "người dân cho biết theo thông tin hôm nay các cơ quan chức năng đã phối hợp tại địa phương".split()

NEWS_TOPICS = ("politics", "accident", "sports", "economy")
NEIGHBOR_PLAN = {
    "Law_and_Government": "law", "Sports": "sports", "Business_and_Industrial": "economy",
    "People_and_Society": "society", "Sensitive_Subjects": "sensitive",
}


def make_text(rng: random.Random, topic: str, length: int = 40) -> str:
    vocab = TOPICS[topic]
    words = [rng.choice(vocab) if rng.random() < 0.7 else rng.choice(FILLER) for _ in range(length)]
    return " ".join(words).capitalize() + "."


def build_records(seed: int = 7, per_news_topic: int = 90, per_neighbor: int = 60) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for topic in NEWS_TOPICS:
        for i in range(per_news_topic):
            rows.append({"id": f"news-{topic}-{i}", "text": make_text(rng, topic), "domain": "News"})
    for label, topic in NEIGHBOR_PLAN.items():
        for i in range(per_neighbor):
            rows.append({"id": f"{label}-{i}", "text": make_text(rng, topic), "domain": label})
    for i in range(30):
        rows.append({"id": f"health-{i}", "text": make_text(rng, "health"), "domain": "Health"})
    # Controlled data-quality defects inside News.
    dup_text = rows[0]["text"]
    rows.append({"id": "news-dup-a", "text": dup_text, "domain": "News"})
    rows.append({"id": "news-dup-b", "text": dup_text.upper().lower().capitalize(), "domain": "News"})
    rows.append({"id": "news-dup-c", "text": "  " + dup_text + "\r\n", "domain": "News"})
    rows.append({"id": "news-empty", "text": "   ", "domain": "News"})
    rows.append({"id": "news-null", "text": None, "domain": "News"})
    rows.append({"id": "news-0-0".replace("0-0", "politics-1"), "text": make_text(rng, "politics"),
                 "domain": "News"})  # duplicated id
    rows.append({"id": "news-url", "text": make_text(rng, "accident") + " Xem https://example.com/a "
                 "hoặc email abc@example.com hotline 0912345678", "domain": "News"})
    rows.append({"id": "other-null-domain", "text": make_text(rng, "health"), "domain": None})
    rng.shuffle(rows)
    return rows


def write_lake(root: Path, rows: list[dict] | None = None, *,
               manifest_name: str = "source_manifest_seed42_n50000.json") -> dict:
    """Create ``00_source_metadata`` + ``01_bronze`` with a correct manifest."""
    rows = rows if rows is not None else build_records()
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    (root / "01_bronze").mkdir(parents=True)
    (root / "00_source_metadata").mkdir(parents=True)
    data = root / "01_bronze" / "bronze_fake.jsonl"
    data.write_bytes(payload)
    manifest = {
        "file_name": data.name, "sample_size": len(rows), "file_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (root / "00_source_metadata" / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest_name": manifest_name, "rows": rows, **manifest}
