#!/bin/bash

MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
SEEDS=(32 42 52)
METHOD_LIST=(avg_confidence min_confidence prompt_ln_nll ft_mmbert_monitoring ft_probe self-reflection gpt_monitoring random_baseline)
OPENAI_API_KEY_PATH=/home/deokhk/research/ZX-seq2seq/nlplab2_openai_key.txt

echo "Starting all tests..."
echo "==== Evaluation on Polymath-low dataset ===="
for i in "${!METHOD_LIST[@]}"; do
    for j in "${!SEEDS[@]}"; do
        METHOD="${METHOD_LIST[$i]}"
        SEED="${SEEDS[$j]}"
        DATASET_TYPE=polymath
        POLYMATH_SPLIT=low
        SIGNAL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/${BASE_MODEL_NAME}/${DATASET_TYPE}_${POLYMATH_SPLIT}/${BASE_MODEL_NAME}_${DATASET_TYPE}_${POLYMATH_SPLIT}_headall_seed_${SEED}_signals_with_label.pth
        TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${POLYMATH_SPLIT}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${POLYMATH_SPLIT}_${SEED}.json
        THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${POLYMATH_SPLIT}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${POLYMATH_SPLIT}_${SEED}_thinking_intv_en.json
        FT_BERT_MDDEL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/mmbert_ft_understandability/${BASE_MODEL_NAME}/mmBERT-base/mgsm_filtered/seed_${SEED}/best_checkpoint
        FT_PROBE_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/probe_ft_understandability/${BASE_MODEL_NAME}/mgsm_filtered/seed_${SEED}/grid_search_best/
        SAVE_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results

        echo "Running method=${METHOD}, seed=${SEED}"
        if [ "$METHOD" == "avg_confidence" ] || [ "$METHOD" == "min_confidence" ] || [ "$METHOD" == "prompt_ln_nll" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --signal_with_label_path ${SIGNAL_PATH} \
                --understandability_test_method ${METHOD} \
                --use_threshold_from_calibration_set \
                --polymath_split ${POLYMATH_SPLIT} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --use_threshold_from_calibration_set \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "random_baseline" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --signal_with_label_path ${SIGNAL_PATH} \
                --understandability_test_method ${METHOD} \
                --polymath_split ${POLYMATH_SPLIT} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "ft_mmbert_monitoring" ]; then
            CUDA_VISIBLE_DEVICES=2 python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --polymath_split ${POLYMATH_SPLIT} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --ft_mmbert_model_path ${FT_BERT_MDDEL_PATH} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "self-reflection" ] || [ "$METHOD" == "gpt_monitoring" ]; then
            CUDA_VISIBLE_DEVICES=2 python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --polymath_split ${POLYMATH_SPLIT} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --gpt_api_key_path ${OPENAI_API_KEY_PATH} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "ft_probe" ]; then
            CUDA_VISIBLE_DEVICES=2 python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --polymath_split ${POLYMATH_SPLIT} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --signal_with_label_path ${SIGNAL_PATH} \
                --ft_probe_model_dir ${FT_PROBE_PATH} \
                --save_dir ${SAVE_DIR}
        fi
    done
done


#!/bin/bash

MODEL_NAME=Qwen/Qwen3-4B
BASE_MODEL_NAME=Qwen3-4B
SEEDS=(32 42 52)
METHOD_LIST=(avg_confidence min_confidence prompt_ln_nll ft_mmbert_monitoring ft_probe self-reflection gpt_monitoring random_baseline)

echo "Starting all tests..."
echo "==== Evaluation on MMLU-ProX-Lite dataset ===="
for i in "${!METHOD_LIST[@]}"; do
    for j in "${!SEEDS[@]}"; do
        METHOD="${METHOD_LIST[$i]}"
        SEED="${SEEDS[$j]}"
        DATASET_TYPE=mmlu_prox_lite
        SIGNAL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_model_signals_with_label/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_${DATASET_TYPE}_headall_seed_${SEED}_signals_with_label.pth
        TASK_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/task_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}.json
        THINK_INTV_EVAL_RESULTS=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results/${BASE_MODEL_NAME}/${DATASET_TYPE}/${BASE_MODEL_NAME}_task_eval_results_${DATASET_TYPE}_${SEED}_thinking_intv_en.json
        FT_BERT_MDDEL_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/mmbert_ft_understandability/${BASE_MODEL_NAME}/mmBERT-base/mmlu_prox_lite_dev/seed_${SEED}/best_checkpoint
        FT_PROBE_PATH=/home/deokhk/research/RLM_analysis/outputs/experiments/probe_ft_understandability/${BASE_MODEL_NAME}/mmlu_prox_lite_dev/seed_${SEED}/grid_search_best/
        SAVE_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results
        echo "Running method=${METHOD}, seed=${SEED}"
        if [ "$METHOD" == "avg_confidence" ] || [ "$METHOD" == "min_confidence" ] || [ "$METHOD" == "prompt_ln_nll" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --signal_with_label_path ${SIGNAL_PATH} \
                --understandability_test_method ${METHOD} \
                --use_threshold_from_calibration_set \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --use_threshold_from_calibration_set \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "random_baseline" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --signal_with_label_path ${SIGNAL_PATH} \
                --understandability_test_method ${METHOD} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "ft_mmbert_monitoring" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --ft_mmbert_model_path ${FT_BERT_MDDEL_PATH} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "self-reflection" ] || [ "$METHOD" == "gpt_monitoring" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --gpt_api_key_path ${OPENAI_API_KEY_PATH} \
                --save_dir ${SAVE_DIR}
        elif [ "$METHOD" == "ft_probe" ]; then
            python -m rlm_analysis.understanding_failure_detection.ut_test \
                --understandability_test_method ${METHOD} \
                --model_name ${MODEL_NAME} \
                --dataset_type ${DATASET_TYPE} \
                --seed ${SEED} \
                --task_eval_results_path ${TASK_EVAL_RESULTS} \
                --thinking_intv_eval_results_path ${THINK_INTV_EVAL_RESULTS} \
                --signal_with_label_path ${SIGNAL_PATH} \
                --ft_probe_model_dir ${FT_PROBE_PATH} \
                --save_dir ${SAVE_DIR}
        fi
    done
done