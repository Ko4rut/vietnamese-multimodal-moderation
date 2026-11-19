"""Generate unapproved local-LLM label suggestions for every Silver V2 News row."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vietnamese_topic_classification.analysis.news_v3 import build_prompt
from vietnamese_topic_classification.config import get_v3_domain_labels
from vietnamese_topic_classification.data.gold.v2 import GoldV2Pipeline
from vietnamese_topic_classification.data.silver import SilverReader

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _parse_response(response: str, allowed_labels: set[str]) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed = None
    for offset, char in enumerate(response):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(response[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise ValueError("Model response did not contain a JSON object")

    label = parsed.get("label")
    confidence = parsed.get("confidence")
    needs_review = parsed.get("needs_review")
    reason = parsed.get("reason")
    if not isinstance(label, str) or label not in allowed_labels:
        raise ValueError("Model response contains an unknown V3 label")
    if type(confidence) not in (int, float) or not 0 <= float(confidence) <= 1:
        raise ValueError("Model response contains an invalid confidence")
    if not isinstance(needs_review, bool):
        raise ValueError("Model response is missing a boolean needs_review field")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Model response is missing a reason")
    return {
        "label": label,
        "confidence": float(confidence),
        "model_suggested_needs_review": needs_review,
        "model_reason": reason.strip()[:500],
    }


def _load_existing(path: Path, allowed_labels: set[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid candidate JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Candidate line {line_number} must be a JSON object")
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"Candidate line {line_number} is missing an id")
            if record_id in results:
                raise ValueError(f"Duplicate candidate id: {record_id}")
            if (
                row.get("approved") is True
                or row.get("needs_review") is False
                or str(row.get("reviewer", "")).strip()
            ):
                raise ValueError(
                    "Refusing to resume over human-reviewed decisions; use a new output file"
                )
            status = row.get("status")
            if status not in {"candidate", "parse_error"}:
                raise ValueError(
                    "Resume file is not a local-model candidate file; use a new output file"
                )
            if status == "candidate":
                if row.get("label") not in allowed_labels:
                    raise ValueError(f"Invalid saved candidate label for {record_id}")
                if type(row.get("confidence")) not in (int, float):
                    raise ValueError(f"Invalid saved candidate confidence for {record_id}")
            results[record_id] = row
    return results


def _write_checkpoint(path: Path, results: dict[str, dict[str, Any]]) -> None:
    checkpoint = path.with_name(path.name + ".checkpointing")
    if checkpoint.exists():
        raise FileExistsError(f"Stale model checkpoint exists: {checkpoint}")
    try:
        with checkpoint.open("x", encoding="utf-8", newline="\n") as handle:
            for record_id in sorted(results):
                handle.write(json.dumps(results[record_id], ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(checkpoint, path)
    finally:
        checkpoint.unlink(missing_ok=True)


def _load_local_model(
    model_name: str,
    model_revision: str,
) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            'Local model dependencies are missing; install with '
            'python -m pip install -e ".[local-llm]"'
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for the default quantized local model. "
            "In Colab, select Runtime > Change runtime type > GPU."
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise RuntimeError(f"Model tokenizer has no chat template: {model_name}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    resolved_revision = str(
        getattr(model.config, "_commit_hash", None) or model_revision
    )
    return torch, model, tokenizer, resolved_revision


def _render_chat(tokenizer: Any, prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer chat template must return text")
    return rendered


def _candidate_row(
    source_row: dict[str, Any],
    model_response: str | None,
    *,
    allowed_labels: set[str],
    model_name: str,
    model_revision: str,
    input_truncated: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": source_row["id"],
        "source_domain": "News",
        "text": source_row["text"],
        "needs_review": True,
        "approved": False,
        "reviewer": "",
        "status": "parse_error",
        "model_name": model_name,
        "model_revision": model_revision,
        "input_truncated": input_truncated,
    }
    if model_response is None:
        row["reason"] = ""
        row["error"] = "Could not parse or validate the local model response"
        return row

    try:
        parsed = _parse_response(model_response, allowed_labels)
    except ValueError as exc:
        row["reason"] = ""
        row["error"] = str(exc)
        return row

    row.update({
        "label": parsed["label"],
        "confidence": parsed["confidence"],
        "model_suggested_needs_review": parsed["model_suggested_needs_review"],
        "reason": parsed["model_reason"],
        "status": "candidate",
    })
    if input_truncated:
        row["reason"] = (
            f'{row["reason"]} [Input was truncated; inspect the full article before approval.]'
        )[:500]
    return row


def run(args: argparse.Namespace) -> None:
    source = SilverReader(args.data_lake_root, args.silver_v2_report)
    if source.report.get("pipeline_version") != "v2":
        raise ValueError("Local News labeling requires a Silver V2 report")
    verified = GoldV2Pipeline(source).verify_silver_publication()
    allowed = set(get_v3_domain_labels())

    news_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in source.iter_records():
        if record.get("domain") != "News":
            continue
        record_id = record.get("id")
        text = record.get("text")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Silver V2 contains a News row with an invalid id")
        if record_id in seen_ids:
            raise ValueError(f"Silver V2 contains a duplicate News id: {record_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Silver V2 News row has invalid text: {record_id}")
        seen_ids.add(record_id)
        news_rows.append(record)
    if not news_rows:
        raise ValueError("Silver V2 contains no News rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"Candidate file already exists; choose --resume or a new output path: {args.output}"
        )
    if args.resume and args.output.exists():
        results = _load_existing(args.output, allowed)
    else:
        results = {}

    unknown_ids = set(results) - seen_ids
    if unknown_ids:
        raise ValueError(f"Resume file contains {len(unknown_ids)} ids outside this Silver V2 source")
    completed = {
        record_id for record_id, row in results.items()
        if row.get("status") == "candidate"
    }
    pending = [row for row in news_rows if row["id"] not in completed]
    if not pending:
        print(json.dumps({
            "status": "already_complete",
            "news_rows": len(news_rows),
            "candidates": len(completed),
            "output": str(args.output),
        }, ensure_ascii=False, indent=2))
        return

    torch, model, tokenizer, resolved_revision = _load_local_model(
        args.model_name,
        args.model_revision,
    )
    labels = get_v3_domain_labels()
    rendered_prompts = [
        _render_chat(
            tokenizer,
            build_prompt(
                row,
                labels,
                max_chars=args.max_text_chars,
            ),
        )
        for row in pending
    ]
    token_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in rendered_prompts
    ]
    batch_count = (len(pending) + args.batch_size - 1) // args.batch_size
    for batch_index, start in enumerate(range(0, len(pending), args.batch_size), 1):
        batch_rows = pending[start:start + args.batch_size]
        batch_prompts = rendered_prompts[start:start + args.batch_size]
        batch_token_lengths = token_lengths[start:start + args.batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
            add_special_tokens=False,
        )
        input_device = model.get_input_embeddings().weight.device
        encoded = encoded.to(input_device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        responses = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        for row, response, token_length in zip(
            batch_rows, responses, batch_token_lengths, strict=True
        ):
            truncated = (
                len(row["text"]) > args.max_text_chars
                or token_length > args.max_input_tokens
            )
            results[row["id"]] = _candidate_row(
                row,
                response,
                allowed_labels=allowed,
                model_name=args.model_name,
                model_revision=resolved_revision,
                input_truncated=truncated,
            )
        if batch_index % args.checkpoint_every == 0 or batch_index == batch_count:
            _write_checkpoint(args.output, results)
            generated_count = sum(
                row.get("status") == "candidate" for row in results.values()
            )
            error_count = sum(
                row.get("status") == "parse_error" for row in results.values()
            )
            print(
                f"Checkpoint {batch_index}/{batch_count}: "
                f"{generated_count} candidates, {error_count} parse errors"
            )

    final_rows = list(results.values())
    counts = {
        "candidates": sum(row.get("status") == "candidate" for row in final_rows),
        "parse_errors": sum(row.get("status") == "parse_error" for row in final_rows),
        "truncated_inputs": sum(bool(row.get("input_truncated")) for row in final_rows),
        "approved": sum(row.get("approved") is True for row in final_rows),
    }
    print(json.dumps({
        "status": "local_suggestions_complete",
        "source_report": source.report_path.name,
        "source_sha256": verified["sha256"],
        "news_rows": len(news_rows),
        **counts,
        "all_candidates_require_human_review": True,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-lake-root", type=Path, required=True)
    parser.add_argument("--silver-v2-report", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-text-chars", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "max_input_tokens", "max_text_chars", "max_new_tokens", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if Path(args.silver_v2_report).name != args.silver_v2_report:
        parser.error("--silver-v2-report must be a report filename, not a path")
    run(args)


if __name__ == "__main__":
    main()
