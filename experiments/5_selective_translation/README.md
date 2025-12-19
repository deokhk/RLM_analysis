# Reproducing Section 5: *Selective Translation for Understanding Failures*

This directory provides scripts and instructions to reproduce the experiments in **Section 5 of the paper**, which evaluates **Selective Translation** as a strategy for mitigating understanding failures.

To evaluate selective translation, we proceed as follows:
1. Measure the performance using **translated queries**.
2. At the **data-point level**, apply an understanding failure detector:
   - If the detector predicts the input is *understood*, use the **Base** inference result.
   - Otherwise, use the inference result from the **translated query**.

The Base (non-translated) performance is assumed to be **already computed**.

---

## 1. Inference with Translated Queries

First, measure model performance when using translated queries.

Run:

```bash
bash 1_inference_translated.sh
```

Please set:
- `TRANSLATED_JSON_PATH`: path to JSON files containing translated questions  
  (translated queries used in the paper—generated with **gpt-4.1**—are provided under `./translated_data`)
- `SAVE_BASE_DIR`: output directory for translated-query inference results

This script performs inference using translated inputs and saves the evaluation results.

---

## 2. Selective Translation Evaluation

Next, evaluate **Selective Translation** using an understanding failure detection method.  
In the paper, we use **`ft_probe`** as the detection method.

Run:

```bash
bash 2_selective_translation.sh
```

Please configure:
- `METHOD=ft_probe`
- `UT_TEST_RESULTS_DIR`: must match the `SAVE_DIR` used in  
  `4_detecting_understanding_failures/5_ut_test.sh`
- `TASK_EVAL_RESULTS`: Base inference results
- `TRANSLATED_EVAL_RESULTS`: path to JSON files under `SAVE_BASE_DIR` generated in Step 1

After execution, files named:

```
selective_translation_ft_probe_${SEED}.csv
```

will be generated.

An example output is shown below:

```csv
scenario,metric,en,de,es,ar,ja,ko,th,bn,sw,te,overall
default,accuracy,0.968,0.88,0.936,0.92,0.856,0.904,0.872,0.864,0.832,0.76,0.8792
default,selective_rate,0.024,0.032,0.056,0.056,0.128,0.016,0.064,0.2,0.864,0.336,0.1776
fnr@0.05,accuracy,0.96,0.88,0.944,0.912,0.872,0.904,0.88,0.888,0.808,0.776,0.8824
fnr@0.05,selective_rate,0.272,0.28,0.288,0.432,0.552,0.192,0.344,0.68,0.912,0.664,0.4616
fnr@0.10,accuracy,0.96,0.88,0.944,0.912,0.872,0.904,0.872,0.872,0.808,0.776,0.88
fnr@0.10,selective_rate,0.104,0.096,0.144,0.112,0.312,0.072,0.192,0.544,0.896,0.544,0.3016
```

---

## Correspondence to Paper Results

- The **accuracy values under the `default` scenario** correspond to the results reported in **Table 4** of the paper.
- Final numbers in Table 4 are obtained by averaging results over **three random seeds**  
  (`32`, `42`, `52`).

---