#!/usr/bin/env python3
"""Record exact Phase 3E train/detection/G0/TF prompt-family equivalence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256, CANONICAL_UNIFIED_QUESTION, CANONICAL_UNIFIED_USER_CONTENT,
    UnifiedForensicsDataset,
)
from eval.forensics_eval import FORENSICS_QUESTION, GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args


def digest(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
    p1 = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text())
    args = build_train_args(p1); tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    rows = [json.loads(line) for line in (ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl").read_text().splitlines() if line]
    fake = next(row for row in rows if int(row["class_label"]) == 1)
    target = UnifiedForensicsDataset._target(fake, "phrase_aligned")
    _, training = UnifiedForensicsDataset._conversation(fake, "phrase_aligned")
    tf = GLaMMForensicsBackend._conversation(target, question=CANONICAL_UNIFIED_QUESTION)
    g0 = GLaMMForensicsBackend._conversation("", question=CANONICAL_UNIFIED_QUESTION)
    detection = GLaMMForensicsBackend._conversation("", question=CANONICAL_UNIFIED_QUESTION)
    legacy = GLaMMForensicsBackend._conversation("", question=FORENSICS_QUESTION)
    token_ids = tokenizer(CANONICAL_UNIFIED_USER_CONTENT, add_special_tokens=False).input_ids
    legacy_ids = tokenizer("The <image> provides an overview of the picture.\n" + FORENSICS_QUESTION,
                           add_special_tokens=False).input_ids
    checks = {
        "config_question_exact": cfg["prompt"]["user_question"] == CANONICAL_UNIFIED_QUESTION,
        "p1_question_exact": p1["forensics"]["canonical_unified_prompt"] == CANONICAL_UNIFIED_QUESTION,
        "training_and_tf_full_conversation_exact": training[0] == tf,
        "detection_and_g0_prompt_exact": detection == g0,
        "canonical_differs_from_legacy": token_ids != legacy_ids,
        "canonical_prompt_hash_exact": digest(CANONICAL_UNIFIED_USER_CONTENT) == CANONICAL_PROMPT_SHA256,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "canonical": {"question": CANONICAL_UNIFIED_QUESTION, "user_content": CANONICAL_UNIFIED_USER_CONTENT,
                            "user_content_sha256": digest(CANONICAL_UNIFIED_USER_CONTENT), "token_ids": token_ids,
                            "token_count": len(token_ids)},
              "P1_training": {"question": CANONICAL_UNIFIED_QUESTION, "assistant_target": target,
                              "full_conversation_sha256": digest(training[0])},
              "Phase3E_detection": {"question": CANONICAL_UNIFIED_QUESTION, "assistant_context": "",
                                    "full_prompt_sha256": digest(detection)},
              "Phase3E_G0": {"question": CANONICAL_UNIFIED_QUESTION, "assistant_context": "",
                             "full_prompt_sha256": digest(g0)},
              "Phase3E_TF_PHRASE": {"question": CANONICAL_UNIFIED_QUESTION, "assistant_context": target,
                                    "full_conversation_sha256": digest(tf)},
              "legacy_localization_question": {"question": FORENSICS_QUESTION, "token_ids": legacy_ids,
                                                "token_count": len(legacy_ids), "full_prompt_sha256": digest(legacy),
                                                "allowed_in_Phase3E": False},
              "interpretation": "User prompt is identical across P1 train and all Phase3E evaluation modes; assistant conditioning differs by mode."}
    dump(ROOT / cfg["experiment"]["output_root"] / "audits/prompt_family_audit.json", result)
    if result["status"] != "PASS": raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
