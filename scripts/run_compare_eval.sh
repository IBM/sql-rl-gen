#!/usr/bin/env bash
set -euo pipefail

# Please activate your Python environment before running (e.g. `conda activate sql-rl-gen`).

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-juierror/flan-t5-text2sql-with-schema-v2}"
TRAINED_AGENT_PATH="${TRAINED_AGENT_PATH:-./output/model_spider_train/best}"
DATASET="${DATASET:-example_text2sql_spider_dev}"
DATASET_NAME="${DATASET_NAME:-spider}"
TEMPLATE="${TEMPLATE:-llama3}"
# Default to >=200 rows to avoid tiny/easy eval sets hiding differences
N_ROWS="${N_ROWS:-200}"
OUT_MD="${OUT_MD:-output/compare_spider_base_vs_rl.md}"

echo "[run_compare_eval.sh] dataset=${DATASET} dataset_name=${DATASET_NAME} n_rows=${N_ROWS}"
echo "[run_compare_eval.sh] model_name_or_path=${MODEL_NAME_OR_PATH}"
echo "[run_compare_eval.sh] trained_agent_path=${TRAINED_AGENT_PATH}"
echo "[run_compare_eval.sh] out_md=${OUT_MD}"

python scripts/run_compare_eval.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --trained_agent_path "${TRAINED_AGENT_PATH}" \
  --dataset "${DATASET}" \
  --dataset_name "${DATASET_NAME}" \
  --template "${TEMPLATE}" \
  --number_of_rows_to_use "${N_ROWS}" \
  --out_md "${OUT_MD}"
