from types import SimpleNamespace

import pytest

from eval.forensics_eval import GLaMMForensicsBackend
from tools.phase3c0 import classify_failure, insert_phrase_tokens, repair_phrase_tokens


class CharacterTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join("[SEG]" if value == 999 else chr(value) for value in ids)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def encoded(text):
    return [999 if part == "[SEG]" else ord(part) for part in text]


def char_ids(text):
    values = []
    index = 0
    while index < len(text):
        if text.startswith("[SEG]", index):
            values.append(999); index += 5
        else:
            values.append(ord(text[index])); index += 1
    return values


def test_phrase_repair_preserves_prefix_and_seg_suffix_exactly():
    tokenizer = CharacterTokenizer()
    original = char_ids("[FAKE] explanation Target regions: wrong arm [SEG] tail")
    result = repair_phrase_tokens(original, tokenizer, 999, "left hand")
    repaired = result["generated_token_ids"]
    assert result["prefix_exact"] and result["seg_suffix_exact"]
    assert tokenizer.decode(repaired) == "[FAKE] explanation Target regions: left hand[SEG] tail"
    assert result["original_phrase_token_ids"] == char_ids(" wrong arm ")


def test_phrase_repair_rejects_missing_or_multiple_seg():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError):
        repair_phrase_tokens(char_ids("Target regions: arm"), tokenizer, 999, "hand")
    with pytest.raises(ValueError):
        repair_phrase_tokens(char_ids("Target regions: arm [SEG] [SEG]"), tokenizer, 999, "hand")


def test_phrase_insertion_does_not_invent_seg():
    tokenizer = CharacterTokenizer()
    original = char_ids("[FAKE] explanation [SEG]")
    result = insert_phrase_tokens(original, tokenizer, 999, "left hand")
    assert result["prefix_exact"] and result["seg_suffix_exact"]
    assert tokenizer.decode(result["generated_token_ids"]) == (
        "[FAKE] explanation  Target regions: left hand[SEG]"
    )
    with pytest.raises(ValueError):
        insert_phrase_tokens(char_ids("[FAKE] explanation"), tokenizer, 999, "left hand")


def test_shared_phase3a_phrase_templates():
    sample = {
        "sample_id": "x", "target_protocol": "phrase_aligned",
        "localization_field": {"normalized_training_phrase": "left hand; right arm"},
        "manifest_row": {"explanation": "  visible   artifact  "},
    }
    assert GLaMMForensicsBackend.phrase_only_content(sample) == (
        "[FAKE] Target regions: left hand; right arm [SEG]"
    )
    assert GLaMMForensicsBackend.tf_phrase_content(sample) == (
        "[FAKE] visible artifact\nTarget regions: left hand; right arm [SEG]"
    )


def test_failure_categories_are_frozen():
    assert classify_failure(.31, .10) == "G0_GOOD"
    assert classify_failure(.30, .70) == "LANGUAGE_RECOVERABLE"
    assert classify_failure(.30, .30) == "PERSISTENT"
    assert classify_failure(.30, .31) == "INTERMEDIATE"
