#!/bin/bash

METHOD=ft_probe
SEED_LIST=(32 42 52 )

MODEL=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B

DATASET_TYPE=polymath
UT_TEST_RESULTS_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results
SAVE_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/selective_translation_${METHOD}

for i in "${!SEED_LIST[@]}"; do
    SEED="${SEED_LIST[$i]}"
    echo "Running selective translation method=${METHOD}, seed=${SEED}"
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/low/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_low_${SEED}.json
    TRANSLATED_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/translated_think_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/low/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_low_${SEED}_translated_think_intv_en.json
    python -m rlm_analysis.selective_translation --understandability_test_method ${METHOD} \
    --model_name ${MODEL} \
    --dataset_type ${DATASET_TYPE} \
    --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te \
    --polymath_split low \
    --task_eval_results_path ${TASK_EVAL_RESULTS} \
    --thinking_intv_eval_results_path ${TRANSLATED_EVAL_RESULTS} \
    --seed ${SEED} \
    --save_dir ${SAVE_DIR} \
    --ut_test_results_dir ${UT_TEST_RESULTS_DIR}
done

DATASET_TYPE=mmlu_prox_lite

for i in "${!SEED_LIST[@]}"; do
    SEED="${SEED_LIST[$i]}"
    echo "Running selective translation method=${METHOD}, seed=${SEED}"
    TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
    TRANSLATED_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/translated_think_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_translated_think_intv_en.json
    python -m rlm_analysis.selective_translation --understandability_test_method ${METHOD} \
    --model_name ${MODEL} \
    --dataset_type ${DATASET_TYPE} \
    --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te \
    --task_eval_results_path ${TASK_EVAL_RESULTS} \
    --thinking_intv_eval_results_path ${TRANSLATED_EVAL_RESULTS} \
    --seed ${SEED} \
    --save_dir ${SAVE_DIR} \
    --ut_test_results_dir ${UT_TEST_RESULTS_DIR}
done