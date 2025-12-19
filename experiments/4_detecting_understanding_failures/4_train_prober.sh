#!/bin/bash


SEED_LIST=(32 42 52 )
DATASET_LIST=(mgsm_filtered mmlu_prox_lite_dev)

OUTPUT_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/probe_ft_understandability
for DATASET_TYPE in "${DATASET_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        SIGNAL_WITH_LABEL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/Qwen3-4B/${DATASET_TYPE}/Qwen3-4B_${DATASET_TYPE}_headall_seed_${SEED}_signals_with_label.pth

        python -m rlm_analysis.understanding_failure_detection.ut_probe_train \
            --signal_with_label_path ${SIGNAL_WITH_LABEL_PATH} \
            --languages "en,de,es,ar,ja,ko,th,bn,sw,te" \
            --llm_model_name Qwen/Qwen3-4B \
            --val_ratio 0.1 \
            --dataset_type ${DATASET_TYPE} \
            --seed ${SEED} \
            --batch_size 16 \
            --epochs 50 \
            --wandb_project ut_probe_training \
            --output_dir ${OUTPUT_DIR}
    done
done