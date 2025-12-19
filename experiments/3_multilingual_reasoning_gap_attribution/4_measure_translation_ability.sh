#!/bin/bash


MODEL_LIST=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "openai/gpt-oss-20b")
OPENAI_KEY_PATH=${YOUR OPENAI_KEY_PATH}
OUTPUT_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/translation_eval_results/

for MODEL in "${MODEL_LIST[@]}"
do
    python -m rlm_analysis.multilingual_reasoning_gap_attribution.eval_translation_ability_gemba_da \
    --model_name $MODEL \
    --output_dir ${OUTPUT_DIR} \
    --max_samples 100 \
    --judge_model gpt-4.1 \
    --openai_key_path $OPENAI_KEY_PATH \
    --gpu_memory_utilization 0.9 \
    --max_concurrent_requests 10 \
    --tensor_parallel_size 1 \
    --max_new_tokens 1024
done