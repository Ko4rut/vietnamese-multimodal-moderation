"""Build Gold V3 train/validation/test splits from a verified Silver V3 snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vietnamese_topic_classification.config import get_v3_domain_labels
from vietnamese_topic_classification.data.gold.config import GoldConfig
from vietnamese_topic_classification.data.gold.exceptions import GoldValidationError
from vietnamese_topic_classification.data.gold.pipeline import GoldPipeline, _sha256_file
from vietnamese_topic_classification.data.gold.validator import GoldValidator
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v3 import SilverV3Pipeline


class GoldV3Pipeline(GoldPipeline):
    """Split Silver V3, keeping label refinement entirely in the Silver stage."""

    version = "v3"

    def __init__(
        self,
        reader: SilverReader,
        *,
        config: GoldConfig | None = None,
        output_root: Path | None = None,
    ) -> None:
        if reader.report.get("pipeline_version") != "v3":
            raise GoldValidationError("Gold V3 requires a Silver V3 quality report")
        super().__init__(reader, config=config, output_root=output_root)

    def _validator(self) -> GoldValidator:
        return GoldValidator(get_v3_domain_labels())

    def _source_metadata(self, verified: dict[str, object]) -> dict[str, Any]:
        source = super()._source_metadata(verified)
        source_report = self.reader.report.get("source")
        if not isinstance(source_report, dict):
            raise GoldValidationError("Silver V3 is missing source lineage")
        source.update({
            "silver_version": "v3",
            "silver_report_sha256": _sha256_file(self.reader.report_path),
            "silver_v2_file": source_report["silver_v2_file"],
            "silver_v2_report": source_report["silver_v2_report"],
            "silver_v2_sha256": source_report["silver_v2_sha256"],
            "silver_v1_report": source_report["silver_v1_report"],
        })
        return source

    def run(self):
        SilverV3Pipeline.verify_reader(self.reader)
        return super().run()
