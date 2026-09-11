"""Phase 6C.2 loss-contract tests for uneven variable-K data parallel batches."""

import torch

from tools.distributed_loss import globally_normalized_components


LOSS_KEYS = ("ce_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss")


def _output(mask_loss):
    zero = mask_loss * 0.0
    return {
        "ce_loss": zero,
        "cls_loss": zero,
        "mask_bce_loss": mask_loss,
        "mask_dice_loss": zero,
    }


def test_uneven_rank_and_gas_mask_loss_is_global_per_mask_mean():
    # [rank][microbatch][mask]. Counts deliberately differ across both axes.
    losses = [
        [[1.0], [2.0, 3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
    ]
    world_size, gas = 2, 2
    global_masks = sum(len(micro) for rank in losses for micro in rank)
    global_counts = {
        "text_tokens": 0,
        "classification_samples": 0,
        "valid_masks": global_masks,
    }

    # DeepSpeed averages the gradients across ranks and accumulation steps.
    dp_gas_averaged = 0.0
    for rank in losses:
        for micro in rank:
            local_mean = torch.tensor(micro, dtype=torch.float64).mean()
            local_counts = {
                "text_tokens": 0,
                "classification_samples": 0,
                "valid_masks": len(micro),
            }
            scaled = globally_normalized_components(
                _output(local_mean), local_counts, global_counts,
                world_size=world_size, accumulation_steps=gas,
            )
            dp_gas_averaged += float(scaled["mask_bce_loss"]) / (world_size * gas)

    expected = sum(value for rank in losses for micro in rank for value in micro) / global_masks
    assert abs(dp_gas_averaged - expected) < 1e-12
    rank_means = sum(
        sum(value for micro in rank for value in micro) / sum(len(micro) for micro in rank)
        for rank in losses
    ) / world_size
    assert dp_gas_averaged != rank_means


def test_uneven_rank_and_gas_gradient_matches_flat_global_objective():
    features = [
        [[1.0], [2.0, 3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
    ]
    world_size, gas = 2, 2
    global_masks = sum(len(micro) for rank in features for micro in rank)
    global_counts = {
        "text_tokens": 0,
        "classification_samples": 0,
        "valid_masks": global_masks,
    }

    theta_distributed = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    scaled_losses = []
    for rank in features:
        for micro in rank:
            x = torch.tensor(micro, dtype=torch.float64)
            local_mean = ((theta_distributed * x - 1.0) ** 2).mean()
            local_counts = {
                "text_tokens": 0,
                "classification_samples": 0,
                "valid_masks": len(micro),
            }
            scaled = globally_normalized_components(
                _output(local_mean), local_counts, global_counts,
                world_size=world_size, accumulation_steps=gas,
            )
            scaled_losses.append(scaled["mask_bce_loss"] / (world_size * gas))
    sum(scaled_losses).backward()

    theta_flat = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    flat = torch.tensor(
        [value for rank in features for micro in rank for value in micro], dtype=torch.float64
    )
    ((theta_flat * flat - 1.0) ** 2).mean().backward()
    torch.testing.assert_close(theta_distributed.grad, theta_flat.grad, rtol=0, atol=1e-12)
