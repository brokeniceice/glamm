import numpy as np

from tools.phase3d0r import (STOPWORDS, average_percentile_ranks, build_idf, content_tokens,
    contradiction_flags, grounding_components, new_rewards, semantic_phrase_score)


def test_stopwords_removed_but_spatial_and_negation_preserved():
    assert content_tokens("The left area with no fingers") == ["left", "no", "fingers"]
    assert "region" in STOPWORDS


def test_idf_specific_token_higher_than_common_token():
    values = build_idf(["hand fingers", "hand eye", "hand button"])
    assert values["button"] > values["hand"]


def test_spatial_contradiction_and_ambiguity():
    assert contradiction_flags("left hand", "right hand")["P_spatial_contra"] == 1
    assert contradiction_flags("left and right hands", "left hand")["AMBIGUOUS_SPATIAL_PHRASE"]


def test_negation_is_record_only_without_scope_parser():
    value = contradiction_flags("hand without fingers", "hand fingers")
    assert value["NEGATION_MISMATCH"] and value["P_neg_contra"] == 0


def test_average_rank_with_ties():
    assert average_percentile_ranks([0.1, 0.1, 0.3]) == [0.25, 0.25, 1.0]


def test_grounding_controllability_suppresses_flat_group():
    values = grounding_components([0.1] * 8)
    assert all(x["C_ground"] == 0 and x["R_ground_rel"] == 0 for x in values)


def test_grounding_controllability_uses_range_and_ceiling():
    values = grounding_components([0.0, 0.3])
    assert values[0]["R_ground_rel"] == 0 and values[1]["R_ground_rel"] == 1


def test_semantic_score_spatial_penalty():
    vectors={"left":np.array([1.,0.]),"right":np.array([1.,0.]),"hand":np.array([0.,1.]),
             "left hand":np.array([1.,1.])/np.sqrt(2),"right hand":np.array([1.,1.])/np.sqrt(2)}
    value=semantic_phrase_score("left hand","right hand",{"left":1,"right":1,"hand":1},vectors)
    assert value["P_spatial_contra"] == 1 and value["R_phrase_sem"] <= .5


def test_new_reward_language_primary_and_real_formula():
    fake={"R_cls":1,"R_struct":1,"R_phrase_sem":.8,"R_ground_rel":.5}
    values=new_rewards(1,fake)
    assert values["Q1"] == .25+.1+.65*.8
    assert values["Q2"] == .25+.1+.55*.8+.1*.5
    assert new_rewards(0,{"R_cls":1,"R_struct":0}) == {"Q1":.75,"Q2":.75,"Q3":.75}
