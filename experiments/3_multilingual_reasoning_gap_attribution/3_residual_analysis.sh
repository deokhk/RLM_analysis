#!/bin/bash

# Set path accordingly
MEAN_JSON_PATH=/home/deokhk/research/RLM_analysis/outputs/multilingual_reasoning_mean.json
STD_JSON_PATH=/home/deokhk/research/RLM_analysis/outputs/multilingual_reasoning_std.json
OUTPUT_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/residual_analysis
N=3

python -m rlm_analysis.multilingual_reasoning_gap_attribution.residual_analysis --mean_json $MEAN_JSON_PATH \
--std_json $STD_JSON_PATH \
--output_dir $OUTPUT_DIR \
--n $N \
--alpha 0.05 \
--compute_significant_language_only