#!/usr/bin/env python3
"""Calibrate CLIP content predictions and build deterministic review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("human", "animal", "object", "scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1",
    )
    parser.add_argument("--target-anchor-precision", type=float, default=0.80)
    parser.add_argument("--random-review-per-category", type=int, default=100)
    parser.add_argument("--low-margin-review", type=int, default=300)
    parser.add_argument("--conflict-review", type=int, default=300)
    parser.add_argument("--view-disagreement-review", type=int, default=300)
    parser.add_argument("--sheet-size", type=int, default=25)
    return parser.parse_args()


def load_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def calibrated_margin_thresholds(
    predictions: Sequence[Mapping[str, Any]], target_precision: float = 0.80
) -> Dict[str, Dict[str, float]]:
    anchors = [
        row for row in predictions
        if row["source"] in {"OpenImagesV7", "COCO2017"}
        and row["source_content_category"] in CATEGORIES
    ]
    thresholds = {}
    for category in CATEGORIES:
        rows = sorted(
            (row for row in anchors if row["predicted_category"] == category),
            key=lambda row: row["top1_top2_margin"],
            reverse=True,
        )
        correct = 0
        best_count = 0
        best_precision = 0.0
        for count, row in enumerate(rows, start=1):
            correct += int(row["source_content_category"] == category)
            precision = correct / count
            if count >= 50 and precision >= target_precision:
                best_count = count
                best_precision = precision
        if best_count:
            threshold = float(rows[best_count - 1]["top1_top2_margin"])
        else:
            correct_margins = [
                row["top1_top2_margin"] for row in rows if row["source_content_category"] == category
            ]
            threshold = float(np.median(correct_margins)) if correct_margins else 1.0
            accepted = [row for row in rows if row["top1_top2_margin"] >= threshold]
            best_count = len(accepted)
            best_precision = (
                sum(row["source_content_category"] == category for row in accepted) / len(accepted)
                if accepted else 0.0
            )
        thresholds[category] = {
            "margin": threshold,
            "anchor_accepted_count": best_count,
            "anchor_precision": best_precision,
            "anchor_predicted_count": len(rows),
        }
    return thresholds


def annotate_predictions(predictions: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> list:
    annotated = []
    for raw in predictions:
        row = dict(raw)
        reasons = []
        category = row["predicted_category"]
        if not row["view_agreement"]:
            reasons.append("view_disagreement")
        if row["top1_top2_margin"] < thresholds[category]["margin"]:
            reasons.append("low_calibrated_margin")
        if row["source"] == "PASS" and category == "human":
            reasons.append("pass_predicted_human")
        source_label = row["source_content_category"]
        if source_label in CATEGORIES and source_label != category:
            reasons.append("source_label_conflict")
        row["review_reasons"] = reasons
        row["review_status"] = "needs_review" if reasons else "auto_accepted"
        annotated.append(row)
    return annotated


def stable_rank(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def select_review_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list:
    selected: Dict[str, Mapping[str, Any]] = {}

    def add(candidates: Iterable[Mapping[str, Any]], limit: int | None = None) -> None:
        values = list(candidates)
        if limit is not None:
            values = values[:limit]
        for row in values:
            selected[row["sample_id"]] = row

    for category in CATEGORIES:
        candidates = sorted(
            (row for row in rows if row["predicted_category"] == category),
            key=lambda row: stable_rank(row["sample_id"]),
        )
        add(candidates, args.random_review_per_category)
    add(sorted(rows, key=lambda row: row["top1_top2_margin"]), args.low_margin_review)
    add(
        sorted(
            (row for row in rows if "source_label_conflict" in row["review_reasons"]),
            key=lambda row: row["top1_top2_margin"],
        ),
        args.conflict_review,
    )
    add(
        sorted(
            (row for row in rows if "view_disagreement" in row["review_reasons"]),
            key=lambda row: row["top1_top2_margin"],
        ),
        args.view_disagreement_review,
    )
    add(row for row in rows if "pass_predicted_human" in row["review_reasons"])
    return sorted(selected.values(), key=lambda row: (row["predicted_category"], stable_rank(row["sample_id"])))


def write_contact_sheets(rows: Sequence[Mapping[str, Any]], output_dir: Path, sheet_size: int) -> list:
    sheet_size = max(1, sheet_size)
    columns = 5
    rows_per_sheet = int(np.ceil(sheet_size / columns))
    cell_width, image_height, label_height = 224, 224, 52
    font = ImageFont.load_default()
    records = []
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for sheet_index, start in enumerate(range(0, len(rows), sheet_size)):
        batch = rows[start : start + sheet_size]
        canvas = Image.new("RGB", (columns * cell_width, rows_per_sheet * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for cell_index, row in enumerate(batch):
            x = (cell_index % columns) * cell_width
            y = (cell_index // columns) * (image_height + label_height)
            try:
                with Image.open(row["image_path"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
                paste_x = x + (cell_width - image.width) // 2
                paste_y = y + (image_height - image.height) // 2
                canvas.paste(image, (paste_x, paste_y))
            except Exception:
                draw.rectangle((x, y, x + cell_width, y + image_height), fill=(80, 0, 0))
            margin = row["top1_top2_margin"]
            text = f"{row['predicted_category']} m={margin:.3f}\n{row['source']} #{row['embedding_index']}"
            draw.text((x + 3, y + image_height + 3), text, fill="black", font=font)
        name = f"review_{sheet_index:03d}.jpg"
        canvas.save(sheets_dir / name, quality=88)
        records.append({
            "sheet": name,
            "sample_ids": [row["sample_id"] for row in batch],
        })
    return records


def nested_counts(rows: Sequence[Mapping[str, Any]], first: str, second: str) -> Dict[str, Dict[str, int]]:
    result = defaultdict(Counter)
    for row in rows:
        result[str(row[first])][str(row[second])] += 1
    return {key: dict(value) for key, value in result.items()}


def main() -> None:
    args = parse_args()
    labels_dir = args.labels_dir.expanduser().resolve()
    predictions = load_jsonl(labels_dir / "predictions.jsonl")
    thresholds = calibrated_margin_thresholds(predictions, args.target_anchor_precision)
    annotated = annotate_predictions(predictions, thresholds)
    review_rows = select_review_rows(annotated, args)

    atomic_write_json(labels_dir / "calibrated_margin_thresholds.json", thresholds)
    atomic_write_jsonl(labels_dir / "predictions_annotated.jsonl", annotated)
    atomic_write_jsonl(labels_dir / "review_candidates.jsonl", review_rows)
    atomic_write_jsonl(
        labels_dir / "pass_predictions.jsonl",
        (row for row in annotated if row["source"] == "PASS"),
    )
    atomic_write_jsonl(
        labels_dir / "synthscars_content_predictions.jsonl",
        (row for row in annotated if row["source"] == "SynthScars"),
    )
    atomic_write_jsonl(
        labels_dir / "real_candidate_predictions.jsonl",
        (row for row in annotated if row["domain"] == "real_candidate"),
    )
    correction_template = [
        {
            "sample_id": row["sample_id"],
            "auto_label": row["predicted_category"],
            "manual_label": None,
            "reviewer_status": "pending",
            "review_reason": row["review_reasons"],
        }
        for row in review_rows
    ]
    atomic_write_jsonl(labels_dir / "manual_corrections_template.jsonl", correction_template)
    sheets = write_contact_sheets(review_rows, labels_dir, args.sheet_size)
    atomic_write_json(labels_dir / "contact_sheet_index.json", sheets)

    confusion = defaultdict(Counter)
    for row in annotated:
        if row["source_content_category"] in CATEGORIES:
            confusion[row["source_content_category"]][row["predicted_category"]] += 1
    summary = {
        "schema_version": "content_labels_analysis_v1",
        "prediction_count": len(annotated),
        "predicted_category_counts": dict(Counter(row["predicted_category"] for row in annotated)),
        "source_predicted_category_counts": nested_counts(annotated, "source", "predicted_category"),
        "domain_predicted_category_counts": nested_counts(annotated, "domain", "predicted_category"),
        "anchor_confusion_source_label_to_clip": {key: dict(value) for key, value in confusion.items()},
        "view_disagreement_count": sum(not row["view_agreement"] for row in annotated),
        "needs_review_count": sum(row["review_status"] == "needs_review" for row in annotated),
        "review_sheet_sample_count": len(review_rows),
        "contact_sheet_count": len(sheets),
        "pass_predicted_human_count": sum(
            row["source"] == "PASS" and row["predicted_category"] == "human" for row in annotated
        ),
        "thresholds": thresholds,
    }
    atomic_write_json(labels_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
