import numpy as np

from tools.phase3d1 import group_relative_advantages, score_q2, score_r3


def test_group_relative_advantages_are_zero_mean_and_order_preserving():
    rewards = [0.1, 0.2, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6]
    advantages = group_relative_advantages(rewards)
    assert abs(sum(advantages)) < 1e-12
    assert np.argmax(advantages) == np.argmax(rewards)
    assert np.argmin(advantages) == np.argmin(rewards)


def test_constant_group_has_no_policy_update_signal():
    assert group_relative_advantages([0.5] * 8) == [0.0] * 8


def test_r3_calls_historical_real_formula():
    rows = [{"R_cls": 1, "R_struct": 0, "R_phrase_lex": 0, "R_mask": 0, "R_align": 0}]
    assert score_r3(0, rows) == [0.7]


def test_q2_real_formula_does_not_call_semantic_or_grounding():
    rows = [{"R_cls": 1, "R_struct": 1, "R_mask": 0, "normalized_phrase": ""} for _ in range(8)]
    rewards, enriched = score_q2(
        0, rows, authoritative_phrase=None, idf={}, embeddings={},
    )
    assert rewards == [1.0] * 8
    assert all(row["R_phrase_sem"] == 0 and row["R_ground_rel"] == 0 for row in enriched)
