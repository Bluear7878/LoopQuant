## Dependencies

```bash
conda create -n loopquant python=3.12
conda activate loopquant
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Prepare models and calibration data

Place each checkpoint so that `model_path` in the YAML resolves to a local Hugging Face directory:

```text
models/Ouro-1.4B
models/Ouro-2.6B
models/Nanbeige4.2-3B
models/huginn-0125
```

By default the loader looks for `models/` in the parent of this repository. To use another location:

```bash
export LOOPQUANT_MODEL_ROOT=/path/to/model_root
```

Calibration uses the WikiText-2 training split (128 sequences, length 2048). Perplexity evaluation downloads WikiText-2 test and a C4 validation shard through `datasets` on first use.

## Post-training quantization

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single.sh configs/ouro_1.4b_w4a4.yaml
```

Other released settings:

```bash
configs/ouro_1.4b_w4a6.yaml
configs/ouro_2.6b_w4a4.yaml
configs/ouro_2.6b_w4a6.yaml
configs/nanbeige_3b_w4a4.yaml
configs/nanbeige_3b_w4a6.yaml
configs/huginn_6step_w4a4.yaml
configs/huginn_6step_w4a6.yaml
```

Each config runs CLDC, ILPA, and LCTC, then the evaluations enabled under `evaluation`. Results are written to the `output_dir` in that config, including `metrics.json`.

## Evaluation

WikiText-2 and C4 perplexity are computed by this repository. Zero-shot QA uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Toggle them in the config:

```yaml
evaluation: {wikitext2: true, c4: true, zero_shot: true, sequence_length: 2048}
```

To evaluate an existing run again:

```bash
python -m loopquant.cli evaluate \
  --model /path/to/Ouro-1.4B \
  --run-dir outputs/ouro_1.4b_w4a4 \
  --wikitext2 --c4 --zero-shot
```

## License

This work is released under the Apache 2.0 license.
