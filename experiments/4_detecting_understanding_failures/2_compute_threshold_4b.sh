#!/bin/bash
set -euo pipefail

MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
SEEDS=(32 42 52)
METHOD_LIST=(avg_confidence min_confidence prompt_ln_nll )
SAVE_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results

for i in "${!METHOD_LIST[@]}"; do
    for j in "${!SEEDS[@]}"; do
        METHOD="${METHOD_LIST[$i]}"
        SEED="${SEEDS[$j]}"
        DATASET_TYPE=mgsm_filtered
        SIGNAL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_${DATASET_TYPE}_headall_seed_${SEED}_signals_with_label.pth
        TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
        THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json
        echo "Running method=${METHOD}, seed=${SEED}"

        python -m rlm_analysis.understanding_failure_detection.ut_test \
            --signal_with_label_path ${SIGNAL_PATH} \
            --understandability_test_method ${METHOD} \
            --model_name ${MODEL_NAME} \
            --dataset_type ${DATASET_TYPE} \
            --seed ${SEED} \
            --task_eval_results_path ${TASK_EVAL_RESULTS} \
            --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
            --save_dir ${SAVE_DIR}

    done
done


for i in "${!METHOD_LIST[@]}"; do
    for j in "${!SEEDS[@]}"; do
        METHOD="${METHOD_LIST[$i]}"
        SEED="${SEEDS[$j]}"
        DATASET_TYPE=mmlu_prox_lite_dev
        SIGNAL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_${DATASET_TYPE}_headall_seed_${SEED}_signals_with_label.pth
        TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
        THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json
        echo "Running method=${METHOD}, seed=${SEED}"

        python -m rlm_analysis.understanding_failure_detection.ut_test \
            --signal_with_label_path ${SIGNAL_PATH} \
            --understandability_test_method ${METHOD} \
            --model_name ${MODEL_NAME} \
            --dataset_type ${DATASET_TYPE} \
            --seed ${SEED} \
            --task_eval_results_path ${TASK_EVAL_RESULTS} \
            --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
            --save_dir ${SAVE_DIR}

    done
done
