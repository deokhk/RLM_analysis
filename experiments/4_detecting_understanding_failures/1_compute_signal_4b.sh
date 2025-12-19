#!/bin/bash


MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
SEED_LIST=(32 42 52)
SAVE_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/


DATASET_TYPE=mgsm_filtered

for SEED in "${SEED_LIST[@]}"; do
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
    THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json

    python -m rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label \
        --model_name ${MODEL_NAME} \
        --task_eval_results_path ${TASK_EVAL_RESULTS} \
        --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
        --batch_size 1 \
        --dataset_type ${DATASET_TYPE} \
        --seed ${SEED} \
        --save_dir ${SAVE_DIR}
done

DATASET_TYPE=mmlu_prox_lite_dev

for SEED in "${SEED_LIST[@]}"; do
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
    THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json

    python -m rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label \
        --model_name ${MODEL_NAME} \
        --task_eval_results_path ${TASK_EVAL_RESULTS} \
        --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
        --batch_size 1 \
        --dataset_type ${DATASET_TYPE} \
        --seed ${SEED} \
        --save_dir ${SAVE_DIR}
done


DATASET_TYPE=polymath
POLYMATH_SPLIT=low

for SEED in "${SEED_LIST[@]}"; do
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${POLYMATH_SPLIT}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${POLYMATH_SPLIT}_${SEED}.json
    THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${POLYMATH_SPLIT}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${POLYMATH_SPLIT}_${SEED}_thinking_intv_en.json

    python -m rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label \
        --model_name ${MODEL_NAME} \
        --task_eval_results_path ${TASK_EVAL_RESULTS} \
        --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
        --batch_size 1 \
        --dataset_type ${DATASET_TYPE} \
        --polymath_split ${POLYMATH_SPLIT} \
        --seed ${SEED} \
        --save_dir ${SAVE_DIR}
done

DATASET_TYPE=mmlu_prox_lite

for SEED in "${SEED_LIST[@]}"; do
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
    THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json

    python -m rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label \
        --model_name ${MODEL_NAME} \
        --task_eval_results_path ${TASK_EVAL_RESULTS} \
        --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
        --batch_size 1 \
        --dataset_type ${DATASET_TYPE} \
        --seed ${SEED} \
        --save_dir ${SAVE_DIR}
done