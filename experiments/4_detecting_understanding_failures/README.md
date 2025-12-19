# Reproducing Section 4: *Detecting Understanding Failures*

This directory provides scripts and instructions to reproduce the experiments in **Section 4, “Detecting Understanding Failures”** of the paper.

---

## 0. Calibration Dataset Inference

Some understanding failure detection methods require a **calibration (or training) dataset**.

- For **Polymath-low**, the calibration dataset is **mgsm_filtered**
- For **MMLU-ProX-Lite**, the calibration dataset is **MMLU-ProX-Lite-dev**

Before computing detection signals, inference must be performed on the calibration dataset.

```bash
bash 0_calibration_dataset_inference_4b.sh
```

This script runs inference on the calibration dataset using the **Qwen/Qwen3-4B** model.

Please set:
- `FASTTEXT_MODEL_PATH`
- `SAVE_BASE_DIR`

according to the instructions in the README of  
`3_multilingual_reasoning_gap_attribution`.

---

## 1. Signal Computation

To train and evaluate understanding failure detection methods, various **signals** (e.g., confidence, input NLL, hidden states) must be computed.

Run:

```bash
bash 1_compute_signal_4b.sh
```

Please configure the following paths appropriately:

- `TASK_EVAL_RESULTS`
- `THINK_INTV_EVAL_RESULTS`
- `SAVE_DIR`

`TASK_EVAL_RESULTS` and `THINK_INTV_EVAL_RESULTS` should point to:
- Outputs from `1_gen_reasoning_trace_with_evaluation_qwen4b.sh` in  
  `3_multilingual_reasoning_gap_attribution`, and
- Outputs from **calibration dataset inference** for calibration data.

After execution, files named `*_signals_with_label.pth` will be generated.  
Each file contains signal values and labels in the following format:

```json
{
  "te": {
    "low-te-20": {
      "avg_confidence": 21.65,
      "min_confidence": 5.06,
      "prompt_ln_nll": 1.47,
      "last_hidden_state": "...",
      "not_understood_label": 1,
      "max_token_to_look_from_reasoning_trace": -1,
      "temperature_used_for_generation": 0.6,
      "confidence_top_k": 20
    }
  }
}
```

---

## 2. Calibration Threshold Computation

For token-probability-based methods (e.g., **Avg Confidence**, **Min Confidence**, **Input NLL**), thresholds must be calibrated using the calibration dataset.

Run:

```bash
bash 2_compute_threshold_4b.sh
```

This script computes thresholds that **maximize F1 score** on the calibration dataset and saves the results as metric summaries.

Please set:
- `SAVE_DIR`
- `SIGNAL_PATH`
- `TASK_EVAL_RESULTS`
- `THINK_INTV_EVAL_RESULTS`

---

## 3. mmBERT Detector Training

The **mmBERT detector** is trained using the calibration dataset:
- `mgsm_filtered` for **Polymath-low**
- `MMLU-ProX-Lite-dev` for **MMLU-ProX-Lite**

Run:

```bash
bash 3_train_mmbert_detector_4b.sh
```

Please configure:
- `TASK_EVAL_RESULTS`
- `THINK_INTV_EVAL_RESULTS`
- `OUTPUT_DIR`

according to the selected `DATASET_TYPE`.

---

## 4. Prober Training

A **Prober** is trained using signals (final hidden states) from the calibration dataset.

Run:

```bash
bash 4_train_prober.sh
```

Please ensure:
- `SIGNAL_WITH_LABEL_PATH` points to the calibration dataset signal file generated in Step 1
- `OUTPUT_DIR` is set appropriately

This script performs a **grid search** and saves the best-performing prober model.

---

## 5. Understanding Failure Testing

To evaluate understanding failure detection methods, run:

```bash
bash 5_ut_test.sh
```

Please configure:
- `SIGNAL_PATH`
- `TASK_EVAL_RESULTS`
- `THINK_INTV_EVAL_RESULTS`
- `FT_BERT_MODEL_PATH`: output directory from Step 3
- `FT_PROBE_PATH`: output directory from Step 4

**Important**:  
`SAVE_DIR` **must be identical** to the `SAVE_DIR` used in **Step 2 (Calibration Threshold Computation)**.  
This is required because `ut_test.py` searches for calibration thresholds by filename matching when  
`--use_threshold_from_calibration_set` is enabled.

---

## 6. Aggregating Results Across Multiple Seeds

To aggregate understanding failure test results across multiple random seeds, run:

```bash
bash 6_ut_aggregate_results.sh
```

Update `RESULTS_DIR` to point to the output directory from Step 5.

After execution, the following files will be generated for each dataset:

- `./aggregated_results/aggregated_overall_metrics_mean.csv`
- `./aggregated_results/aggregated_overall_metrics_std.csv`

These results correspond to the values reported in **Table 3** of the paper.