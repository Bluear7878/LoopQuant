#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
if [[ $# -ne 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_single.sh configs/<run>.yaml" >&2
  exit 2
fi
config="$1"
if [[ ! -f "${config}" ]]; then
  echo "config not found: ${config}" >&2
  exit 2
fi
read -r model_path output_dir <<<"$(python - "${config}" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(config["model_path"], config["output_dir"])
PY
)"
if [[ -n "${LOOPQUANT_MODEL_ROOT:-}" ]]; then
  model_root="${LOOPQUANT_MODEL_ROOT}"
elif [[ -d "${ROOT}/../loopquant/models" ]]; then
  model_root="${ROOT}/../loopquant"
else
  model_root="${ROOT}/.."
fi
if [[ "${model_path}" != /* ]]; then
  model_path="${model_root}/${model_path}"
fi
mkdir -p "${output_dir}"
echo "[LoopQuant] START config=${config} model=${model_path}" | tee -a "${output_dir}/run.log"
set +e
python -m loopquant.cli quantize \
  --model "${model_path}" \
  --config "${config}" \
  --output-dir "${output_dir}" \
  --evaluate-wikitext2 \
  2>&1 | tee -a "${output_dir}/run.log"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[LoopQuant] FAILED config=${config} status=${status}" | tee -a "${output_dir}/run.log"
  exit "${status}"
fi
echo "[LoopQuant] DONE config=${config}" | tee -a "${output_dir}/run.log"
