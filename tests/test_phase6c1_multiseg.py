import torch
import pytest

from dataset.dataset import _truncate_training_batch_preserving_seg
from model.GLaMM import (
    calculate_dice_loss,
    compute_sigmoid_cross_entropy,
    extract_seg_predictor_hidden,
)
from model.pcerf import multiseg_flatten, multiseg_regroup, multiseg_r1_batch


def test_k1_predictor_backward_parity():
    seg, image = 99, -200
    ids = torch.tensor([[1, image, 7, seg, 8]])
    hidden = torch.randn(1, ids.shape[1] + 575, 8, requires_grad=True)
    grouped, positions = extract_seg_predictor_hidden(hidden, ids, seg)
    direct = hidden[0, positions[0]]
    assert torch.equal(grouped[0], direct)
    grouped[0].sum().backward()
    first_gradient = hidden.grad.clone()
    hidden.grad.zero_()
    direct.sum().backward()
    assert torch.equal(hidden.grad, first_gradient)


def test_k2_and_variable_k_order_offsets():
    grouped = [torch.tensor([[10.0]]), torch.tensor([[20.0], [21.0]]), torch.tensor([[30.0], [31.0], [32.0]])]
    flat, image_index, offsets = multiseg_flatten(grouped)
    assert flat[:, 0].tolist() == [10, 20, 21, 30, 31, 32]
    assert image_index.tolist() == [0, 1, 1, 2, 2, 2]
    assert offsets.tolist() == [0, 1, 3, 6]
    assert all(torch.equal(a, b) for a, b in zip(grouped, multiseg_regroup(flat, offsets)))


def test_r1_batch_is_flatten_repeat_regroup_without_parameters():
    q = [torch.randn(1, 256), torch.randn(2, 256), torch.randn(3, 256)]
    z = [torch.randn(1, 1, 8, 8), torch.randn(2, 1, 8, 8), torch.randn(3, 1, 8, 8)]
    images = {
        "S64": torch.randn(3, 2, 4, 4),
        "F24": torch.randn(3, 2, 3, 3),
        "z_F24": torch.randn(3, 1, 3, 3),
        "clip_geometries": [{"id": index} for index in range(3)],
    }
    batch, offsets = multiseg_r1_batch(q, z, images)
    assert batch["q_seg"].shape[0] == sum(x.shape[0] for x in q) == 6
    assert offsets.tolist() == [0, 1, 3, 6]
    assert batch["S64"][:, 0, 0, 0].tolist() == [images["S64"][i, 0, 0, 0].item() for i in (0,1,1,2,2,2)]
    assert [x["id"] for x in batch["clip_geometries"]] == [0,1,1,2,2,2]


def test_k1_native_global_per_mask_loss_backward_parity():
    pred_old = torch.randn(1, 8, 8, requires_grad=True)
    pred_new = pred_old.detach().clone().requires_grad_(True)
    target = torch.randint(0, 2, (1, 8, 8)).float()
    def loss(pred):
        return (2.0 * compute_sigmoid_cross_entropy(pred, target, 1)
                + 0.5 * calculate_dice_loss(pred, target, 1))
    old, new = loss(pred_old), loss(pred_new)
    assert torch.equal(old, new)
    old.backward(); new.backward()
    assert torch.equal(pred_old.grad, pred_new.grad)


def test_global_per_mask_mean_remains_native():
    p1, p2 = torch.randn(1, 8, 8), torch.randn(2, 8, 8)
    t1, t2 = torch.rand_like(p1), torch.rand_like(p2)
    combined_p, combined_t = torch.cat((p1, p2)), torch.cat((t1, t2))
    native = compute_sigmoid_cross_entropy(combined_p, combined_t, 3)
    manual = sum(compute_sigmoid_cross_entropy(p, t, p.shape[0]) * p.shape[0] for p,t in ((p1,t1),(p2,t2))) / 3
    assert torch.allclose(native, manual)


def test_native_multiseg_generic_truncation_fails_closed():
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return type("Tokens", (), {"input_ids": [9] if text == "[SEG]" else [1, 2]})()
    ids = torch.tensor([[1, 2, 3, 9, 4, 9]])
    with pytest.raises(ValueError, match="atomically pair-truncated"):
        _truncate_training_batch_preserving_seg(
            ids, ids.clone(), torch.ones_like(ids), Tokenizer(), 5,
            target_protocols=["native_multiseg"],
            multiseg_pair_counts=[2],
        )


def test_native_multiseg_k0_uses_legacy_prefix_truncation():
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return type("Tokens", (), {"input_ids": [9] if text == "[SEG]" else [1, 2]})()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    truncated, labels, attention, _ = _truncate_training_batch_preserving_seg(
        ids, ids.clone(), torch.ones_like(ids), Tokenizer(), 5,
        target_protocols=["native_multiseg"], multiseg_pair_counts=[0],
    )
    assert truncated.tolist() == [[1, 2, 3, 4, 5]]
    assert labels.tolist() == [[1, 2, 3, 4, 5]]
    assert attention.tolist() == [[True, True, True, True, True]]
