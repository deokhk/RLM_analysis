#!/bin/bash

MODEL_NAME=Qwen/Qwen3-4B
EVAL_LANGS="en,de,es,ar,ja,ko,th,bn,sw,te"

SEEDS=(32 42 52)
DATASETS=("polymath" "mmlu_prox_lite")

SAVE_BASE_DIR=/home/deokhk/research/RLM_analysis/outputs/translated_think_intv_eval_results
for SEED in "${SEEDS[@]}"; do
  for DATASET_TYPE in "${DATASETS[@]}"; do

    if [ "${DATASET_TYPE}" == "polymath" ]; then
      TRANSLATED_JSON_PATH=/home/deokhk/research/RLM_analysis/translated_data/polymath/polymath_low_translated_to_english.json
    elif [ "${DATASET_TYPE}" == "mmlu_prox_lite" ]; then
      TRANSLATED_JSON_PATH=/home/deokhk/research/RLM_analysis/translated_data/mmlu-prox-lite/mmlu_prox_lite_test_translated_to_english.json
    else
      echo "Unknown dataset type: ${DATASET_TYPE}"
      exit 1
    fi

    echo "Running: SEED=${SEED}, DATASET=${DATASET_TYPE}"

    python -m rlm_analysis.generate_reasoning_trace_with_evaluation \
      --model_name "${MODEL_NAME}" \
      --dataset_type "${DATASET_TYPE}" \
      --eval_langs "${EVAL_LANGS}" \
      --seed "${SEED}" \
      --test_with_translated_data_as_ut \
      --translated_dataset_json_path "${TRANSLATED_JSON_PATH}" \
      --save_base_dir "${SAVE_BASE_DIR}"

  done
done
