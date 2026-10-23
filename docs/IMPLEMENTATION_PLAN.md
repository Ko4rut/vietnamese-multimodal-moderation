# Implementation plan

| Phase | Scope | Acceptance criteria |
| --- | --- | --- |
| 1 | Repo, schema, taxonomy, packaging, CI | Offline unit tests run; editable install works |
| 2 | Text MVP | Versioned splits; baselines, macro-F1 and per-label recall |
| 3 | Image | OCR + visual encoder; validated image test cases |
| 4 | Video | ASR + sampled frames + temporal evidence |
| 5 | Fusion & policy | Compare text-only, visual-only and fused metrics; human-review rules |
| 6 | Demo & evaluation | FastAPI/UI, reproducibility report, documented limitations |

## Next tasks

1. Review label definitions with annotators and version the annotation manual.
2. Verify dataset licenses and de-duplicate train/validation/test sources.
3. Implement model and dataset adapters behind the shared inference contract.
4. Evaluate ambiguous educational, journalistic and quoted content.
5. Keep moderation and safety-critical high-risk review human-governed.
