#!/usr/bin/env python3
"""Freeze Phase 3D.0-R inputs, encoder, preprocessing, corpus and IDF."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from tools.phase3b_replay import file_sha256
from tools.phase3d0r import CURATED_SYNONYMS, NEGATIONS, SPATIAL_PAIRS, STOPWORDS, build_idf

CFG_PATH = ROOT / "configs/phase3d0r_reward_reformulation.yaml"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text()); out = (ROOT / cfg["experiment"]["output_root"]).resolve()
    source = (ROOT / cfg["experiment"]["source_root"]).resolve(); manifest = (ROOT / cfg["experiment"]["manifest_dir"]).resolve()
    for name in ("audit", "semantic_encoder", "phrase_corpus", "idf", "reward_components", "group_grounding",
                 "reward_candidates", "reward_dev", "validation", "subgroups", "statistics", "qualitative", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    completion = load(source / "completion_manifest.json")
    if completion["status"] != "COMPLETE" or not completion["P1_weights_exact"] or not completion["no_training"]:
        raise RuntimeError("Phase 3D.0 source is not a complete frozen artifact")
    source_files = [source / "rollouts/dev_A_text_shard00_of_01.jsonl",
                    source / "rollouts/val_A_full_shard00_of_02.jsonl", source / "rollouts/val_A_full_shard01_of_02.jsonl",
                    source / "greedy/val_shard00_of_02.jsonl", source / "greedy/val_shard01_of_02.jsonl",
                    source / "manifests/reward_dev_manifest.json", source / "manifests/reward_val_manifest.json",
                    source / "audit/mask_provenance.json", source / "sampling_protocol.json"]
    if any(not path.exists() for path in source_files): raise FileNotFoundError([str(x) for x in source_files if not x.exists()])
    source_artifact = {
        "status": "FROZEN_SOURCE_VERIFIED", "source_root": str(source),
        "files": [{"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size} for path in source_files],
        "validation_groups": 2212, "validation_trajectories": 17696, "K": 8,
        "reward_dev_groups": 1024, "reward_dev_text_trajectories": 8192,
        "reward_dev_full_mask_groups": 1,
        "reward_dev_grounding_available_in_source": False,
        "reward_dev_gap": "Phase 3D.0 reward-dev A is text-only; only one separate integration-smoke group has masks",
        "user_authorized_resolution": "fixed-token no-cache mask replay for reward-dev only; no rerollout",
        "no_new_rollout_generation": True, "new_mask_scope": "reward-dev frozen-token replay only",
    }
    dump(out / "phase3d0_source_artifact.json", source_artifact)

    train = rows(manifest / "train_combined.jsonl")
    phrases = [UnifiedForensicsDataset.authoritative_localization_field(row)["normalized_training_phrase"]
               for row in train if int(row["class_label"]) == 1]
    idf = build_idf(phrases)
    corpus_payload = {"status": "FROZEN_BEFORE_REWARD_DEV", "source": str(manifest / "train_combined.jsonl"),
                      "source_sha256": file_sha256(manifest / "train_combined.jsonl"), "documents": len(phrases),
                      "unique_phrases": len(set(phrases)), "phrases": phrases,
                      "validation_or_test_used": False}
    dump(out / "phrase_corpus/authoritative_train_phrases.json", corpus_payload)
    dump(out / "token_idf.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "formula": "log((N+1)/(df+1))+1 normalized by corpus maximum",
                                   "documents": len(phrases), "values": idf})
    dump(out / "idf/token_idf.json", load(out / "token_idf.json"))
    dump(out / "stopword_list.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "values": sorted(STOPWORDS),
                                       "negations_explicitly_preserved": sorted(NEGATIONS)})
    dump(out / "critical_modifier_lexicon.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "spatial_antonym_pairs": SPATIAL_PAIRS,
                                                   "weight_multiplier": 2.0, "negations": sorted(NEGATIONS)})
    dump(out / "contradiction_rules.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "spatial_pairs": SPATIAL_PAIRS,
                                             "spatial_penalty": 0.5, "negation_mismatch_recorded": True,
                                             "negation_penalty_applied": False,
                                             "reason": "scope is not inferred without deterministic dependency parsing",
                                             "blanket_entity_antonyms": False})
    dump(out / "phrase_preprocessing.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "lowercase": True,
                                               "unicode_normalization": "NFKC", "punctuation_removed": True,
                                               "whitespace_normalized": True, "lemmatization": False,
                                               "lemmatizer_reason": "no deterministic lemmatizer installed",
                                               "stopwords_removed": True, "spatial_preserved": True, "negation_preserved": True})
    enc = cfg["semantic_encoder"]; cache = Path(enc["cache_dir"])
    model_files = sorted(path for path in cache.rglob("*") if path.is_file() and ".lock" not in str(path))
    if not model_files: raise RuntimeError("SEMANTIC_ENCODER_UNAVAILABLE")
    dump(out / "semantic_encoder_provenance.json", {"status": "FROZEN_BEFORE_REWARD_DEV", **enc,
          "backend": "transformers AutoModel; sentence-transformers compatible masked-mean pooling",
          "eval": True, "requires_grad": False, "fine_tuned": False,
          "files": [{"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)} for path in model_files]})
    dump(out / "reward_definition.json", {"status": "FROZEN_BEFORE_REWARD_DEV", "OLD_R3": "HISTORICAL_REWARD_CONTROL",
          "R_phrase_lex": "LEXICAL_DIAGNOSTIC_ONLY", "R_align": "OLD_R_ALIGN_DIAGNOSTIC_ONLY",
          "R_phrase_sem": "clip((.65*R_key_soft+.35*R_sentence_sem)*(1-.5*P_spatial)*(1-.5*P_neg),0,1)",
          "R_ground_rel": "C_range*C_ceiling*average_percentile_rank(mask_iou)",
          "candidates": cfg["reward"], "weights_frozen_before_validation": True, "validation_sweep": False})
    dump(out / "reward_dev/selection_protocol.json", {
        "status": "FROZEN_BEFORE_REWARD_DEV_SCORING", "preferred_candidate": "Q2",
        "Q2_conditions": {
            "lm_accuracy_delta_vs_Q1_min": -0.005, "structure_delta_vs_Q1_min": -0.005,
            "phrase_sem_delta_vs_Q1_min": -0.02,
            "high_controllability_mask_delta_vs_Q1_min": 0.0,
            "low_controllability_phrase_delta_vs_Q1_min": -0.02,
            "each_proxy_failure_count_not_above_OLD_R3": True,
            "aggregate_proxy_failure_relative_reduction_min": 0.10,
            "aggregate_small_count_rule": "if OLD_R3 total <10 require at least one fewer failure",
        },
        "failure_proxies": ["LEXICAL_HIGH_SEMANTIC_LOW", "PHRASE_GOOD_MASK_CEILING_PUNISHED", "SPATIAL_CONTRADICTION_SELECTED"],
        "Q3_exploratory_only": True, "validation_formula_or_threshold_changes_allowed": False,
    })
    provenance = {"phase": "Phase 3D.0-R — Evidence-Aware Reward Reformulation", "status": "PREPARED",
                  "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "read_only_offline_rescoring": True, "config": str(CFG_PATH), "config_sha256": file_sha256(CFG_PATH),
                  "checkpoint_sha256_expected": cfg["source"]["checkpoint_sha256"], "model_loaded": False,
                  "optimizer_created": False, "backward_called": False, "new_rollout_generation": False,
                  "reward_dev_fixed_token_mask_replay_user_authorized": True,
                  "new_mask_generation_scope": "reward-dev existing token IDs only", "phase3d1_started": False}
    dump(out / "provenance.json", provenance)
    print(json.dumps({"status": "PREPARED", "output": str(out), "train_phrase_documents": len(phrases),
                      "reward_dev_grounding_available_in_source": False,
                      "reward_dev_mask_replay_authorized": True}, indent=2))


if __name__ == "__main__": main()
