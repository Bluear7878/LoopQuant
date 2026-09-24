"""Perplexity and optional zero-shot evaluation for a live fake-quantized model."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset


@torch.no_grad()
def calibration_nll(model, sequences: torch.Tensor | list[torch.Tensor], device: str) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    for sequence in sequences:
        input_ids = sequence.unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tokens += int(labels.numel())
    mean = total_nll / max(total_tokens, 1)
    return {"nll": mean, "ppl": math.exp(mean), "tokens": total_tokens}


def _tokenize_wikitext2_test(tokenizer) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer(
        "\n\n".join(dataset["text"]), add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]


def _tokenize_c4_validation(tokenizer) -> torch.Tensor:
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-validation.00000-of-00008.json.gz",
        split="train",
    )
    return tokenizer(" ".join(dataset[:1100]["text"]), return_tensors="pt").input_ids[0][: 256 * 2048]


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    device: str = "cuda:0",
    sequence_length: int = 2048,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate full fixed windows using the recurrent model forward path."""
    if dataset.lower() in {"wikitext2", "wikitext-2"}:
        tokens = _tokenize_wikitext2_test(tokenizer)
        name = "wikitext2"
    elif dataset.lower() == "c4":
        tokens = _tokenize_c4_validation(tokenizer)
        name = "c4"
    else:
        raise ValueError("dataset must be wikitext2 or c4")
    windows = math.ceil(tokens.numel() / int(sequence_length))
    if limit is not None:
        windows = min(windows, int(limit))
    if windows == 0:
        raise RuntimeError("evaluation corpus does not contain one complete window")
    total_nll = 0.0
    total_targets = 0
    evaluated_windows = 0
    for index in range(windows):
        window = tokens[index * sequence_length : (index + 1) * sequence_length]
        if window.numel() < 2:
            continue
        batch = window.unsqueeze(0).to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss)
        total_targets += int(batch.shape[1] - 1)
        evaluated_windows += 1
    mean_nll = total_nll / total_targets
    return {
        "dataset": name,
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "windows": evaluated_windows,
        "sequence_length": int(sequence_length),
        "target_tokens": total_targets,
        "formal_full_dataset": limit is None,
    }


def evaluate_zero_shot(
    model,
    tokenizer,
    device: str,
    tasks: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run the optional paper suite through lm-evaluation-harness."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise RuntimeError("zero-shot evaluation requires the optional lm-eval package") from error
    tasks = tasks or [
        "arc_challenge", "arc_easy", "hellaswag", "lambada_openai",
        "lambada_standard", "piqa", "winogrande",
    ]
    adapter = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size, trust_remote_code=True)
    result = evaluator.simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
        limit=limit,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
    )
    return {"tasks": tasks, "limit": limit, "results": result.get("results", {})}
