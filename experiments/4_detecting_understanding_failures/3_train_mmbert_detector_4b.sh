#!/bin/bash


# Train using "mgsm-filtered"
SEED_LIST=(32 42 52 )
EVAL_LANGS=en,de,es,ar,ja,ko,th,bn,sw,te

MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
DATASET_TYPE=mgsm_filtered
OUTPUT_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/mmbert_ft_understandability
for SEED in "${SEED_LIST[@]}"; do
  TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
  THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json

  python -m rlm_analysis.understanding_failure_detection.ut_mmbert_monitoring_train \
    --model_name jhu-clsp/mmBERT-base \
    --task_eval_results_path ${TASK_EVAL_RESULTS} \
    --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
    --eval_langs ${EVAL_LANGS} \
    --dataset_type ${DATASET_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --fp16 \
    --seed ${SEED} \
    --llm_model_name ${MODEL_NAME}
done 


# Train using "mmlu-prox-lite-dev"
SEED_LIST=(32 42 52 )
EVAL_LANGS=en,de,es,ar,ja,ko,th,bn,sw,te

MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
DATASET_TYPE=mmlu_prox_lite_dev
OUTPUT_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/mmbert_ft_understandability
for SEED in "${SEED_LIST[@]}"; do
  TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
  THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json

  python -m rlm_analysis.understanding_failure_detection.ut_mmbert_monitoring_train \
    --model_name jhu-clsp/mmBERT-base \
    --task_eval_results_path ${TASK_EVAL_RESULTS} \
    --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
    --eval_langs ${EVAL_LANGS} \
    --dataset_type ${DATASET_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --fp16 \
    --seed ${SEED} \
    --llm_model_name ${MODEL_NAME}
done 
