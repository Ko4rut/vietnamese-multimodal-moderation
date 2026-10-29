"""Post-fit diagnostics for sklearn text baselines (no access to Gold test)."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from vietnamese_topic_classification.evaluation.metrics import calculate_metrics
from vietnamese_topic_classification.training.model import TextSplit


def _write_figure(fig: object, output: Path) -> None:
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_train_validation(train: dict, validation: dict, destination: Path) -> None:
    metrics = ("accuracy", "macro_f1", "weighted_f1")
    names = ("Accuracy", "Macro-F1", "Weighted-F1")
    positions = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - 0.19 for x in positions], [train[k] for k in metrics], width=0.38, label="Train")
    ax.bar([x + 0.19 for x in positions], [validation[k] for k in metrics], width=0.38, label="Validation")
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Train vs validation (post-fit diagnostic, NOT epoch history)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _write_figure(fig, destination)


def _plot_per_class(labels: Sequence[str], matrix: list[list[int]], f1: dict, directory: Path) -> list[dict]:
    actual = [sum(row) for row in matrix]
    correct = [matrix[i][i] for i in range(len(labels))]
    recall = [correct[i] / actual[i] if actual[i] else 0.0 for i in range(len(labels))]
    rank = sorted(range(len(labels)), key=lambda i: (-correct[i], labels[i]))
    fig, ax = plt.subplots(figsize=(13, 10))
    positions = list(range(len(labels)))
    ax.barh(positions, [correct[i] for i in rank], label="Correct")
    ax.barh(positions, [actual[i] - correct[i] for i in rank], left=[correct[i] for i in rank], label="Incorrect")
    ax.set_yticks(positions, [labels[i] for i in rank])
    ax.invert_yaxis()
    ax.set_xlabel("Validation samples (count)")
    ax.set_title("Correct predictions by true class (count; affected by class support)")
    ax.legend()
    _write_figure(fig, directory / "correct_counts_by_class.png")

    recall_rank = sorted(range(len(labels)), key=lambda i: (-recall[i], labels[i]))
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.barh(positions, [100 * recall[i] for i in recall_rank])
    ax.set_yticks(positions, [labels[i] for i in recall_rank])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recall (%)")
    ax.set_title("Recall by true class (fairer than raw correct counts)")
    _write_figure(fig, directory / "recall_by_class.png")

    f1_rank = sorted(labels, key=lambda label: f1[label])
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.barh(positions, [f1[label] for label in f1_rank])
    ax.set_yticks(positions, f1_rank)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("Per-class validation F1 (lowest at top)")
    ax.invert_yaxis()
    _write_figure(fig, directory / "per_class_f1.png")

    normalized = [
        [value / sum(row) if sum(row) else 0 for value in row]
        for row in matrix
    ]
    fig, ax = plt.subplots(figsize=(15, 13))
    heatmap = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Validation confusion matrix (normalized by true class)")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, label="Fraction")
    _write_figure(fig, directory / "confusion_matrix_normalized.png")

    return [
        {"class": labels[i], "support": actual[i], "correct": correct[i], "recall": recall[i], "f1": f1[labels[i]]}
        for i in range(len(labels))
    ]


def _plot_learning_curve(points: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [100 * p["train_fraction"] for p in points]
    ax.plot(xs, [p["train_macro_f1"] for p in points], "o-", label="Train Macro-F1")
    ax.plot(xs, [p["validation_macro_f1"] for p in points], "o-", label="Validation Macro-F1")
    ax.set(xlabel="Fraction of Gold train used (%)", ylabel="Macro-F1",
           title="Learning curve by training set size (NOT by epoch)", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    _write_figure(fig, path)


def save_diagnostics(
    pipeline: object,
    train: TextSplit,
    validation: TextSplit,
    *,
    label_names: Sequence[str],
    output_dir: Path,
    fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    seed: int = 42,
) -> dict:
    """Write fresh training diagnostics once; no model mutation and no Gold test access.

    Learning-curve points refit CLONES on stratified subsets of Gold train.
    These are separate fits, not recorded epochs of the original model.
    """
    if output_dir.exists():
        raise FileExistsError(f"Diagnostics already exist: {output_dir}")
    if not fractions or tuple(sorted(set(fractions))) != fractions or fractions[-1] != 1.0:
        raise ValueError("fractions must be sorted, unique, positive and end at 1.0")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("Learning curve fractions must be in (0, 1]")
    if not train["texts"] or not validation["texts"]:
        raise ValueError("Train and validation must not be empty")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".diagnostics-staging-", dir=output_dir.parent))
    try:
        train_predictions = pipeline.predict(train["texts"])
        validation_predictions = pipeline.predict(validation["texts"])
        train_report = calculate_metrics(train["labels"], train_predictions, label_names=label_names)
        validation_report = calculate_metrics(
            validation["labels"], validation_predictions, label_names=label_names
        )
        train_summary = {k: train_report[k] for k in ("accuracy", "macro_f1", "weighted_f1")}
        validation_summary = {
            k: validation_report[k] for k in ("accuracy", "macro_f1", "weighted_f1")
        }
        gap = {k: train_summary[k] - validation_summary[k] for k in train_summary}
        print(f"[diagnostics] Full train: {train_summary}", flush=True)
        print(f"[diagnostics] Validation: {validation_summary}; gap: {gap}", flush=True)
        _plot_train_validation(train_summary, validation_summary, staging / "train_vs_validation.png")
        per_class = _plot_per_class(
            label_names, validation_report["confusion_matrix"], validation_report["per_class_f1"],
            staging,
        )

        points = []
        for fraction in fractions:
            started = time.perf_counter()
            if fraction == 1.0:
                local_train = train_summary
                local_validation = validation_summary
                size = len(train["labels"])
            else:
                subset, _ = train_test_split(
                    list(range(len(train["labels"]))), train_size=fraction,
                    stratify=train["labels"], random_state=seed,
                )
                fitted = clone(pipeline)
                fitted.fit([train["texts"][i] for i in subset], [train["labels"][i] for i in subset])
                local_train_report = calculate_metrics(
                    [train["labels"][i] for i in subset],
                    fitted.predict([train["texts"][i] for i in subset]),
                    label_names=label_names,
                )
                local_validation_report = calculate_metrics(
                    validation["labels"], fitted.predict(validation["texts"]),
                    label_names=label_names,
                )
                local_train, local_validation = local_train_report, local_validation_report
                size = len(subset)
                del fitted
            point = {
                "train_fraction": fraction,
                "train_size": size,
                "train_macro_f1": local_train["macro_f1"],
                "validation_macro_f1": local_validation["macro_f1"],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            points.append(point)
            print(
                f"[diagnostics] Train size {size:,} ({fraction:.0%}) | "
                f"train Macro-F1={point['train_macro_f1']:.4f} | "
                f"validation Macro-F1={point['validation_macro_f1']:.4f} | "
                f"time={point['elapsed_seconds']:.1f}s", flush=True,
            )
        _plot_learning_curve(points, staging / "learning_curve_by_train_size.png")
        estimator = pipeline.named_steps["classifier"]
        n_iter = getattr(estimator, "n_iter_", None)
        iterations = n_iter.tolist() if hasattr(n_iter, "tolist") else n_iter
        report = {
            "note": "Solver iterations are not epochs; learning curve points are independent fits.",
            "train_rows": len(train["labels"]),
            "validation_rows": len(validation["labels"]),
            "train_metrics": train_summary,
            "validation_metrics": validation_summary,
            "train_minus_validation_gap": gap,
            "solver_iterations": iterations,
            "learning_curve": points,
            "per_class": per_class,
        }
        (staging / "diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Diagnostics already exist: {output_dir}")
        staging.rename(output_dir)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
