"""Interaction-Aware Loop-wise Precision Allocation (ILPA)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from .evaluation import calibration_nll
from .quantization import QUANTIZATION_CONTEXT, QuantizedLinear, fake_quantize_activation


def enumerate_precision_allocations(
    num_loops: int,
    candidate_bits: Sequence[int] = (4, 8),
    average_bits: float = 6.0,
) -> list[tuple[int, ...]]:
    """Enumerate loop allocations satisfying the exact average-bit budget."""
    candidates = tuple(int(bit) for bit in candidate_bits)
    target = float(average_bits) * int(num_loops)
    return [
        allocation
        for allocation in itertools.product(candidates, repeat=int(num_loops))
        if abs(sum(allocation) - target) < 1e-8
    ]


def compute_loop_marginal_sensitivity(
    coalition_scores: dict[tuple[int, ...], float],
    candidate_bits: Sequence[int] = (4, 8),
    monte_carlo_samples: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the cost of reducing each loop's precision over loop permutations.

    Fewer than five loops uses every permutation. Larger loop counts estimate
    the same average from ``monte_carlo_samples`` permutations.
    """
    low, high = min(candidate_bits), max(candidate_bits)
    num_loops = len(next(iter(coalition_scores)))
    players = tuple(range(num_loops))
    permutations = list(itertools.permutations(players))
    if num_loops >= 5:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        count = min(int(monte_carlo_samples), len(permutations))
        chosen = torch.randperm(len(permutations), generator=generator)[:count]
        permutations = [permutations[int(index)] for index in chosen]
    rows = []
    for permutation in permutations:
        allocation = [high] * num_loops
        marginal = torch.zeros(num_loops, dtype=torch.float64)
        for player in permutation:
            before = tuple(allocation)
            allocation[player] = low
            after = tuple(allocation)
            marginal[player] = float(coalition_scores[after] - coalition_scores[before])
        rows.append(marginal)
    matrix = torch.stack(rows)
    return matrix.mean(0), matrix


def compute_loop_interactions(
    marginal_samples: torch.Tensor,
    shrinkage: float = 0.5,
) -> torch.Tensor:
    """Estimate non-additive loop interactions from marginal-cost co-variation."""
    centered = marginal_samples - marginal_samples.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(marginal_samples.shape[0], 1)
    diagonal = torch.diag(torch.diag(covariance))
    return (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal


def select_ilpa_allocation(
    num_loops: int,
    candidate_bits: Sequence[int],
    average_bits: float,
    marginal_sensitivity: torch.Tensor,
    interactions: torch.Tensor,
) -> tuple[tuple[int, ...], dict[tuple[int, ...], float]]:
    """Select the fixed-budget allocation using sensitivity and interaction."""
    low = min(int(bit) for bit in candidate_bits)
    candidates = enumerate_precision_allocations(num_loops, candidate_bits, average_bits)
    if not candidates:
        raise ValueError("no activation allocation satisfies the requested average-bit budget")
    scores: dict[tuple[int, ...], float] = {}
    for allocation in candidates:
        reduced = torch.tensor([1.0 if bit == low else 0.0 for bit in allocation], dtype=torch.float64)
        scores[allocation] = float(marginal_sensitivity @ reduced + reduced @ interactions @ reduced)
    return min(scores, key=scores.get), scores


def _fit_bit_matched_clip_bank(
    modules: dict[str, QuantizedLinear],
    statistics: dict[str, dict[int, torch.Tensor]],
    candidate_bits: Sequence[int],
    ratios: Sequence[float],
) -> dict[tuple[str, int, int], float]:
    bank: dict[tuple[str, int, int], float] = {}
    for name, module in modules.items():
        for loop, raw in statistics.get(name, {}).items():
            transformed = module.transform_activation(raw.to(module.weight.device, module.weight.dtype)).float()
            for bits in candidate_bits:
                best_ratio, best_error = 1.0, float("inf")
                for ratio in ratios:
                    reconstruction = fake_quantize_activation(transformed, int(bits), module.group_size, float(ratio))
                    error = float((reconstruction - transformed).square().mean())
                    if error < best_error:
                        best_ratio, best_error = float(ratio), error
                bank[(name, int(loop), int(bits))] = best_ratio
    return bank


def _apply_clip_bank(
    modules: dict[str, QuantizedLinear],
    allocation: Sequence[int],
    bank: dict[tuple[str, int, int], float],
) -> None:
    for name, module in modules.items():
        module.set_clip_ratios(
            [bank.get((name, loop, int(bits)), 1.0) for loop, bits in enumerate(allocation)]
        )


@dataclass
class InteractionAwareLoopwisePrecisionAllocation:
    """Allocate activation precision using calibration-only loop effects."""

    num_loops: int
    candidate_bits: tuple[int, ...] = (4, 8)
    average_bits: float = 6.0
    interaction_shrinkage: float = 0.5
    selection_samples: int = 32
    clip_ratios: tuple[float, ...] = tuple(round(0.50 + 0.01 * index, 2) for index in range(51))

    def run(
        self,
        model,
        calibration_data: torch.Tensor,
        cross_loop_statistics: dict[str, dict[int, torch.Tensor]],
        device: str,
        score_function: Callable = calibration_nll,
    ) -> dict[str, object]:
        modules = {name: module for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}
        setting_is_uniform = len(self.candidate_bits) == 1 or self.average_bits in self.candidate_bits
        if setting_is_uniform and all(bit == int(self.average_bits) for bit in self.candidate_bits):
            allocation = tuple([int(self.average_bits)] * self.num_loops)
            QUANTIZATION_CONTEXT.set_allocation(allocation)
            QUANTIZATION_CONTEXT.enabled = True
            return {
                "num_loops": self.num_loops,
                "candidate_bits": list(self.candidate_bits),
                "average_activation_bits": self.average_bits,
                "allocation": list(allocation),
                "selection": "unique fixed-budget allocation",
                "evaluation_data_used": False,
            }
        bank = _fit_bit_matched_clip_bank(modules, cross_loop_statistics, self.candidate_bits, self.clip_ratios)
        selection_data = calibration_data[-min(self.selection_samples, len(calibration_data)) :]
        coalition_scores: dict[tuple[int, ...], float] = {}
        QUANTIZATION_CONTEXT.enabled = True
        for allocation in itertools.product(self.candidate_bits, repeat=self.num_loops):
            QUANTIZATION_CONTEXT.set_allocation(allocation)
            _apply_clip_bank(modules, allocation, bank)
            result = score_function(model, selection_data, device)
            coalition_scores[tuple(allocation)] = float(result["nll"])
        sensitivity, samples = compute_loop_marginal_sensitivity(
            coalition_scores, self.candidate_bits, seed=0
        )
        interactions = compute_loop_interactions(samples, self.interaction_shrinkage)
        allocation, fixed_budget_scores = select_ilpa_allocation(
            self.num_loops,
            self.candidate_bits,
            self.average_bits,
            sensitivity,
            interactions,
        )
        QUANTIZATION_CONTEXT.set_allocation(allocation)
        _apply_clip_bank(modules, allocation, bank)
        return {
            "num_loops": self.num_loops,
            "candidate_bits": list(self.candidate_bits),
            "average_activation_bits": self.average_bits,
            "allocation": list(allocation),
            "selection": "interaction-aware fixed-budget calibration selection",
            "evaluation_data_used": False,
            "coalition_scores": {"".join(map(str, key)): value for key, value in coalition_scores.items()},
            "marginal_sensitivity": sensitivity.tolist(),
            "interactions": interactions.tolist(),
            "fixed_budget_scores": {"".join(map(str, key)): value for key, value in fixed_budget_scores.items()},
        }
