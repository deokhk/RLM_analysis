#!/bin/bash

#!/bin/bash

MODEL_NAME=Qwen/Qwen3-4B
SEED_LIST=(32 42 52)
FASTTEXT_MODEL_PATH=/home/deokhk/research/RLM_analysis/misc/lid.176.ftz

SAVE_BASE_DIR=/home/deokhk/research/RLM_analysis/outputs/task_eval_results
# Generate reasoning trace and evaluate performance
for SEED in "${SEED_LIST[@]}"; do
    python -m rlm_analysis.generate_reasoning_trace_with_evaluation --model_name ${MODEL_NAME} \
    --dataset_type mgsm_filtered --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te --seed ${SEED} \
    --fasttext_model_path ${FASTTEXT_MODEL_PATH} --save_base_dir ${SAVE_BASE_DIR}

    python -m rlm_analysis.generate_reasoning_trace_with_evaluation --model_name ${MODEL_NAME} \
    --dataset_type mmlu_prox_lite_dev --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te --seed ${SEED} \
    --fasttext_model_path ${FASTTEXT_MODEL_PATH} --save_base_dir ${SAVE_BASE_DIR}
done

# Generate reasoning trace with thinking (understanding) intervention 
SAVE_BASE_DIR=/home/deokhk/research/RLM_analysis/outputs/thinking_intv_eval_results

for SEED in "${SEED_LIST[@]}"; do
    python -m rlm_analysis.generate_reasoning_trace_with_evaluation --model_name ${MODEL_NAME} \
    --dataset_type mgsm_filtered --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te --seed ${SEED} \
    --fasttext_model_path ${FASTTEXT_MODEL_PATH} --save_base_dir ${SAVE_BASE_DIR} --do_thinking_intervention

    python -m rlm_analysis.generate_reasoning_trace_with_evaluation --model_name ${MODEL_NAME} \
    --dataset_type mmlu_prox_lite_dev --eval_langs en,de,es,ar,ja,ko,th,bn,sw,te --seed ${SEED} \
    --fasttext_model_path ${FASTTEXT_MODEL_PATH} --save_base_dir ${SAVE_BASE_DIR} --do_thinking_intervention
done