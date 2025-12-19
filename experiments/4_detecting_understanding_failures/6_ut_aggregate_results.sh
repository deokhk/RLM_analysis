#!/bin/bash

RESULTS_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results/Qwen3-4B/polymath_low
python -m rlm_analysis.understanding_failure_detection.ut_aggregate_results --results_dir ${RESULTS_DIR} \
--metrics_using_threshold_from_calibration_set 

RESULTS_DIR=/home/deokhk/research/RLM_analysis/outputs/experiments/ut_test_results/Qwen3-4B/mmlu_prox_lite
python -m rlm_analysis.understanding_failure_detection.ut_aggregate_results --results_dir ${RESULTS_DIR} \
--metrics_using_threshold_from_calibration_set