"""GLaMM-ready wrapper for the frozen unified Real/Fake forensic manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_IMAGE_TOKEN

from .synthscars import polygon_to_mask, polygons_for_target


REAL_TOKEN = "[REAL]"
FAKE_TOKEN = "[FAKE]"
SEG_TOKEN = "[SEG]"
TARGET_PROTOCOL_HISTORICAL = "historical"
TARGET_PROTOCOL_PHRASE_ALIGNED = "phrase_aligned"
PHRASE_FIELD_PREFIX = "Target regions:"
REAL_EXPLANATION = "No identifiable synthetic artifact evidence is detected."
CANONICAL_UNIFIED_QUESTION = "Determine whether this image is authentic and explain the forensic evidence."
CANONICAL_UNIFIED_USER_CONTENT = (
    f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n{CANONICAL_UNIFIED_QUESTION}"
)
CANONICAL_PROMPT_TEMPLATE_ID = "unified_forensics_v1"
CANONICAL_PROMPT_SHA256 = hashlib.sha256(
    CANONICAL_UNIFIED_USER_CONTENT.encode("utf-8")
).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


class UnifiedForensicsDataset(torch.utils.data.Dataset):
    """Read a frozen split and emit the standard GLaMM sample mapping.

    Real and Fake samples pass through the exact same global and grounding
    image preprocessing. Only Fake samples carry a mask and ``seg_valid``.
    ``[CLS]`` is deliberately inserted by the collator at the assistant prefix.
    """

    IMG_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(
        self,
        manifest_dir: str | Path,
        tokenizer,
        global_image_encoder: str,
        *,
        split: str = "train",
        datasets_root: str | Path = "datasets",
        synthscars_root: str | Path | None = None,
        image_size: int = 1024,
        global_enc_processor=None,
        transform=None,
        target_protocol: str = TARGET_PROTOCOL_HISTORICAL,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported unified forensic split: {split}")
        self.manifest_dir = Path(manifest_dir).expanduser().resolve()
        self.datasets_root = Path(datasets_root).expanduser().resolve()
        self.synthscars_root = (
            Path(synthscars_root).expanduser().resolve()
            if synthscars_root is not None
            else self.datasets_root / "SynthScars"
        )
        self.tokenizer = tokenizer
        self.split = split
        if target_protocol not in {
            TARGET_PROTOCOL_HISTORICAL, TARGET_PROTOCOL_PHRASE_ALIGNED,
        }:
            raise ValueError(f"Unsupported target protocol: {target_protocol}")
        self.target_protocol = target_protocol
        self.rows = _load_jsonl(self.manifest_dir / f"{split}_combined.jsonl")
        self.global_enc_processor = global_enc_processor or CLIPImageProcessor.from_pretrained(
            global_image_encoder
        )
        self.transform = transform or ResizeLongestSide(image_size)

        for index, row in enumerate(self.rows):
            domain = row.get("forensics_domain")
            label = row.get("class_label")
            if domain not in {"real", "fake"} or label not in {0, 1}:
                raise ValueError(
                    f"{self.manifest_dir}/{split}_combined.jsonl row {index}: "
                    f"invalid domain/label {domain!r}/{label!r}"
                )
            expected = 1 if domain == "fake" else 0
            if int(label) != expected:
                raise ValueError(f"row {index}: domain {domain} conflicts with class_label={label}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, row: Mapping[str, Any]) -> Path:
        candidate = row.get("image_path")
        if candidate and Path(candidate).is_file():
            return Path(candidate).expanduser().resolve()
        root = self.synthscars_root if row["forensics_domain"] == "fake" else self.datasets_root
        path = root / str(row["image_relpath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @classmethod
    def grounding_enc_processor(cls, image: torch.Tensor) -> torch.Tensor:
        image = (image - cls.IMG_MEAN) / cls.IMG_STD
        height, width = image.shape[-2:]
        if height > cls.IMG_SIZE or width > cls.IMG_SIZE:
            raise ValueError(f"grounding image exceeds {cls.IMG_SIZE}: {(height, width)}")
        return F.pad(image, (0, cls.IMG_SIZE - width, 0, cls.IMG_SIZE - height))

    @staticmethod
    def _fake_union_mask(row: Mapping[str, Any], height: int, width: int) -> torch.Tensor:
        refs: Sequence[Mapping[str, Any]] = row.get("refs") or []
        if not refs:
            raise ValueError(f"Fake sample {row.get('sample_id')} has no evidence refs")
        union = np.zeros((height, width), dtype=bool)
        for ref in refs:
            polygons = polygons_for_target(ref, height, width)
            union |= polygon_to_mask(polygons, height, width).astype(bool)
        if not union.any():
            raise ValueError(f"Fake sample {row.get('sample_id')} has an empty evidence mask")
        return torch.from_numpy(union.astype(np.float32)).unsqueeze(0)

    @staticmethod
    def authoritative_localization_field(row: Mapping[str, Any]) -> dict[str, Any]:
        """Construct the single-SEG localization field without paraphrasing refs."""
        if row["forensics_domain"] != "fake":
            return {
                "raw_phrases": [], "normalized_training_phrase": None,
                "construction_rule": "real_historical_protocol_unchanged",
            }
        raw_phrases = [" ".join(str(ref.get("phrase") or "").split()) for ref in row.get("refs") or []]
        if not raw_phrases or any(not phrase for phrase in raw_phrases):
            raise ValueError(f"Fake sample {row.get('sample_id')} lacks an authoritative phrase")
        deduplicated = []
        for phrase in raw_phrases:
            if phrase not in deduplicated:
                deduplicated.append(phrase)
        return {
            "raw_phrases": raw_phrases,
            "normalized_training_phrase": "; ".join(deduplicated),
            "construction_rule": (
                "annotation_order_whitespace_normalized_exact_duplicate_deduplicated_semicolon_join"
            ),
        }

    @staticmethod
    def _target(
        row: Mapping[str, Any], target_protocol: str = TARGET_PROTOCOL_HISTORICAL
    ) -> str:
        if row["forensics_domain"] == "real":
            return f"{REAL_TOKEN} {REAL_EXPLANATION}"
        explanation = " ".join(str(row.get("explanation") or "").split())
        if not explanation:
            raise ValueError(f"Fake sample {row.get('sample_id')} has no explanation")
        if target_protocol == TARGET_PROTOCOL_HISTORICAL:
            return f"{FAKE_TOKEN} {explanation} {SEG_TOKEN}"
        if target_protocol != TARGET_PROTOCOL_PHRASE_ALIGNED:
            raise ValueError(f"Unsupported target protocol: {target_protocol}")
        phrase = UnifiedForensicsDataset.authoritative_localization_field(row)[
            "normalized_training_phrase"
        ]
        return f"{FAKE_TOKEN} {explanation}\n{PHRASE_FIELD_PREFIX} {phrase} {SEG_TOKEN}"

    @staticmethod
    def _conversation(
        row: Mapping[str, Any], target_protocol: str = TARGET_PROTOCOL_HISTORICAL
    ) -> tuple[list[str], list[str]]:
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], CANONICAL_UNIFIED_USER_CONTENT)
        conv.append_message(conv.roles[1], UnifiedForensicsDataset._target(row, target_protocol))
        return [CANONICAL_UNIFIED_QUESTION], [conv.get_prompt()]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve_image_path(row)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"OpenCV failed to decode {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        global_enc_image = self.global_enc_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        resized_image = self.transform.apply_image(image)
        resize = resized_image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(
            torch.from_numpy(resized_image).permute(2, 0, 1).contiguous()
        )

        seg_valid = row["forensics_domain"] == "fake"
        masks = self._fake_union_mask(row, original_height, original_width) if seg_valid else None
        questions, conversations = self._conversation(row, self.target_protocol)
        if self.target_protocol == TARGET_PROTOCOL_PHRASE_ALIGNED:
            localization_field = self.authoritative_localization_field(row)
        else:
            # Historical Phase 2A does not consume refs.phrase.  Keeping this
            # path phrase-agnostic preserves compatibility with old manifests
            # and test fixtures that legitimately omit that optional field.
            localization_field = {
                "raw_phrases": [], "normalized_training_phrase": None,
                "construction_rule": "historical_protocol_does_not_consume_phrase",
            }
        label = torch.full(
            (original_height, original_width), self.IGNORE_LABEL, dtype=torch.long
        )
        return {
            "image_path": str(image_path),
            "global_enc_image": global_enc_image,
            "grounding_enc_image": grounding_enc_image,
            "bboxes": None,
            "conversations": conversations,
            "masks": masks,
            "label": label,
            "resize": resize,
            "questions": questions,
            "sampled_classes": ["synthetic artifact"] if seg_valid else [],
            "cls_label": int(row["class_label"]),
            "seg_valid": seg_valid,
            "sample_id": row["sample_id"],
            "source": row["source"],
            "content_category": row.get("content_category"),
            "manifest_row": dict(row),
            "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256,
            "target_protocol": self.target_protocol,
            "localization_field": localization_field,
        }
