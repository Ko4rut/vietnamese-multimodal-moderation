"""Offline smoke test of the Bronze reader entry point."""

import hashlib
import json

from vietnamese_topic_classification.data.main import MANIFEST_NAME, main


def test_data_main_reads_config_and_snapshot(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "01_bronze"
    source_dir.mkdir()
    raw = b'{"id":"1","text":"Xin chao","domain":"News"}\n'
    (source_dir / "sample.jsonl").write_bytes(raw)

    metadata = tmp_path / "00_source_metadata"
    metadata.mkdir()
    (metadata / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "file_name": "sample.jsonl",
                "sample_size": 1,
                "file_size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_LAKE_ROOT", str(tmp_path))
    main()

    output = capsys.readouterr().out
    assert "BRONZE VALIDATION PASSED" in output
    assert "Records: 1" in output
    assert "First record domain: News" in output
