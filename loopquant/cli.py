"""Command-line interface for the public LoopQuant pipeline."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from .config import LoopQuantConfig
from .data import load_calibration_data
from .evaluation import evaluate_perplexity, evaluate_zero_shot
from .models import load_looped_transformer
from .pipeline import LoopQuantPipeline
from .utils import (
    resolve_device,
    runtime_metadata,
    set_random_seed,
    setup_logging,
    write_json,
    write_yaml,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Local Ouro checkpoint directory")
    parser.add_argument("--model-family", default=None, choices=("ouro", "nanbeige", "huginn"))
    parser.add_argument("--config", required=True, help="LoopQuant YAML configuration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", help="Device such as cuda:0, cpu, or auto")
    parser.add_argument("--weight-bits", type=int)
    parser.add_argument("--activation-setting", choices=("w4a4", "w4a6"))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--num-calibration-samples", type=int)
    parser.add_argument("--calibration-seqlen", type=int)
    parser.add_argument("--calibration-seed", type=int)
    parser.add_argument("--evaluate-wikitext2", action="store_true")
    parser.add_argument("--evaluate-c4", action="store_true")
    parser.add_argument("--evaluate-zero-shot", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopquant", description="LoopQuant for Looped Transformers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    quantize = subparsers.add_parser("quantize", help="Run CLDC, ILPA, LCTC, and optional evaluation")
    _add_common_arguments(quantize)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Reconstruct a configured fake-quantized run and evaluate it",
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--wikitext2", action="store_true")
    evaluate.add_argument("--c4", action="store_true")
    evaluate.add_argument("--zero-shot", action="store_true")
    return parser


def _load_resolved_config(args: argparse.Namespace) -> LoopQuantConfig:
    config = LoopQuantConfig.from_yaml(args.config)
    config.apply_overrides(
        model_family=args.model_family,
        output_dir=args.output_dir,
        quantization__weight_bits=args.weight_bits,
        quantization__activation_setting=args.activation_setting,
        quantization__group_size=args.group_size,
        calibration__num_samples=args.num_calibration_samples,
        calibration__sequence_length=args.calibration_seqlen,
        calibration__seed=args.calibration_seed,
        evaluation__wikitext2=True if args.evaluate_wikitext2 else None,
        evaluation__c4=True if args.evaluate_c4 else None,
        evaluation__zero_shot=True if args.evaluate_zero_shot else None,
    )
    return config


def _run_quantize(args: argparse.Namespace) -> int:
    started = time.time()
    config = _load_resolved_config(args)
    output = Path(config.output_dir).expanduser().resolve()
    logger = setup_logging(output)
    device = resolve_device(args.device)
    set_random_seed(int(config.calibration["seed"]))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    logger.info("[LoopQuant] Model: %s", config.model_variant or Path(args.model).name)
    logger.info("[LoopQuant] Setting: %s", str(config.quantization["activation_setting"]).upper())
    logger.info(
        "[LoopQuant] Calibration: WikiText-2 train, %d x %d, seed %d",
        int(config.calibration["num_samples"]),
        int(config.calibration["sequence_length"]),
        int(config.calibration["seed"]),
    )
    write_yaml(output / "config.yaml", config.to_dict())
    model, tokenizer = load_looped_transformer(
        args.model,
        model_family=config.model_family,
        device=device,
        num_loops=int(config.num_loops),
    )
    calibration_data = load_calibration_data(
        tokenizer,
        dataset=config.calibration["dataset"],
        split=config.calibration["split"],
        num_samples=int(config.calibration["num_samples"]),
        sequence_length=int(config.calibration["sequence_length"]),
        seed=int(config.calibration["seed"]),
    )
    if tuple(calibration_data.shape) != (
        int(config.calibration["num_samples"]),
        int(config.calibration["sequence_length"]),
    ):
        raise RuntimeError(f"unexpected calibration shape {tuple(calibration_data.shape)}")
    pipeline = LoopQuantPipeline(config, device=device, logger=logger)
    result = pipeline.run(model, tokenizer, calibration_data)
    logger.info("[ILPA] Selected allocation: %s", result.allocation)
    logger.info("[LCTC] Calibration complete")
    write_json(output / "cldc_summary.json", result.cldc)
    ilpa_output = {
        **result.ilpa,
        "activation_setting": str(config.quantization["activation_setting"]).lower(),
        "allocation": result.allocation,
    }
    write_json(output / "ilpa_allocation.json", ilpa_output)
    write_json(output / "lctc_summary.json", result.lctc)
    metrics: dict[str, object] = {}
    if config.evaluation.get("wikitext2"):
        logger.info("[Evaluation] Running full WikiText-2 test perplexity")
        metrics["wikitext2"] = evaluate_perplexity(
            result.model, tokenizer, "wikitext2", device=device, sequence_length=2048
        )
        logger.info("[Evaluation] WikiText-2 PPL: %.6f", metrics["wikitext2"]["perplexity"])
    if config.evaluation.get("c4"):
        logger.info("[Evaluation] Running C4 perplexity")
        metrics["c4"] = evaluate_perplexity(result.model, tokenizer, "c4", device=device)
    if config.evaluation.get("zero_shot"):
        logger.info("[Evaluation] Running zero-shot suite")
        metrics["zero_shot"] = evaluate_zero_shot(
            result.model,
            tokenizer,
            device,
            batch_size=int(config.evaluation.get("batch_size", 1)),
        )
    for value in metrics.values():
        if isinstance(value, dict) and "perplexity" in value and not math.isfinite(float(value["perplexity"])):
            raise FloatingPointError("evaluation produced non-finite perplexity")
    write_json(output / "metrics.json", metrics)
    metadata = runtime_metadata(started)
    if device.startswith("cuda"):
        metadata["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
    summary = {
        "model": str(args.model),
        "model_family": config.model_family,
        "model_variant": config.model_variant,
        "num_loops": int(config.num_loops),
        "weight_bits": int(config.quantization["weight_bits"]),
        "weight_group_size": int(config.quantization["group_size"]),
        "activation_setting": config.quantization["activation_setting"],
        "activation_allocation": result.allocation,
        "fake_quantization": True,
        "calibration": {
            "dataset": "wikitext2_train",
            "num_samples": int(calibration_data.shape[0]),
            "sequence_length": int(calibration_data.shape[1]),
            "seed": int(config.calibration["seed"]),
        },
        "stage_order": list(result.stage_order),
        "runtime": metadata,
    }
    write_json(output / "quantization_summary.json", summary)
    logger.info("[LoopQuant] Output: %s", output)
    logger.info("[LoopQuant] Runtime: %.1f seconds", metadata["runtime_seconds"])
    if metadata.get("peak_cuda_memory_gb") is not None:
        logger.info("[LoopQuant] Peak CUDA memory: %.2f GiB", metadata["peak_cuda_memory_gb"])
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"run configuration not found: {config_path}")
    requested = args.wikitext2 or args.c4 or args.zero_shot
    replay = argparse.Namespace(
        model=args.model,
        model_family="ouro",
        config=str(config_path),
        output_dir=str(run_dir / "evaluation_replay"),
        device=args.device,
        weight_bits=None,
        activation_setting=None,
        group_size=None,
        num_calibration_samples=None,
        calibration_seqlen=None,
        calibration_seed=None,
        evaluate_wikitext2=args.wikitext2 or not requested,
        evaluate_c4=args.c4,
        evaluate_zero_shot=args.zero_shot,
    )
    return _run_quantize(replay)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "quantize":
        return _run_quantize(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
