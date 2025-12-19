# Reproducing Section 3: *Why Does the Multilingual Reasoning Gap Emerge?*

This recipe provides step-by-step instructions to reproduce the experiments in **Section 3, “Why Does the Multilingual Reasoning Gap Emerge?”** of the paper.

---

## 1. Generate Reasoning Traces and Evaluation Results

First, run the script below to generate reasoning traces and evaluation results for the **Qwen3-4B** model. If you want to reproduce experiments on other models,
you can modify the given script:

```bash
bash 1_gen_reasoning_trace_with_evaluation_qwen4b.sh
```

Before running the script, please ensure that the following variables are correctly set:

- `FASTTEXT_MODEL_PATH`: path to the FastText language identification model.
- `SAVE_BASE_DIR`: base directory for saving outputs.

**Important**:
- For the **Base setting** (without any intervention), `SAVE_BASE_DIR` **must end with**:
  ```
  outputs/task_eval_results
  ```
- For the **Understanding Intervention** setting, `SAVE_BASE_DIR` **must end with**:
  ```
  outputs/thinking_intv_eval_results
  ```

As a result:
- Base (no intervention) results will be saved under `./task_eval_results/`
- Understanding Intervention results will be saved under `./thinking_intv_eval_results/`

---

## 2. Aggregate Multilingual Reasoning Results

Next, aggregate the results from both settings by running 2_aggregate_result.sh:

```bash
OUTPUT_DIR=/home/deokhk/research/RLM_dump/outputs
python -m rlm_analysis.multilingual_reasoning_gap_attribution.aggregate_multilingual_reasoning_result   --outputs_dir ${OUTPUT_DIR}
```

Here, `OUTPUT_DIR` should be set to the **`outputs/` directory that contains the `SAVE_BASE_DIR` specified in Step 1**.

After execution, the following files will be generated:

- `multilingual_reasoning_mean.json`
- `multilingual_reasoning_std.json`
Below is a sample of multilingual_reasoning_mean.json
```json
{
  "Qwen3-4B": {
    "MMLU-ProX-Lite": {
      "Base": {
        "en": 77.0428,
        "de": 77.82100000000001,
        "es": 76.524,
        "ar": 73.28146666666667,
        "ja": 74.9676,
        "ko": 74.5785,
        "th": 73.92999999999999,
        "bn": 74.5785,
        "sw": 53.5668,
        "te": 71.07650000000001,
        "Avg": 72.73671666666667
      },
      "w/ T": {
        "en": ...,
```

These outputs are used to reproduce:
- **Table 1**: Accuracy comparison across languages
- **Table 2**: Average reasoning performance ratio (using the `Avg` field)

---

## 3. Residual Analysis (Figure 2)

To perform the residual analysis described in the paper, run:

```bash
bash 3_residual_analysis.sh
```

Please set:
- `MEAN_JSON_PATH`, `STD_JSON_PATH`: paths to the outputs generated in Step 2
- `OUTPUT_DIR`: directory for saving residual analysis results
- `n`: number of random seeds used in Step 1 (the paper uses **n = 3**)

After execution, the file below will be generated:

```
failure_attribution_shares_siglangonly.json
```

Below is a sample of the file.
```json
{
  "Qwen3-4B": {
    "MMLU-ProX-Lite": {
      "per_lang": {
        "ar": {
          "shares": {
            "U": 1.0,
            "G": 0.0,
            "R": 0.0
        },
        ...
      },
      "aggregate": {
        "Avg_unweighted": {
          "U": 0.9763411249622806,
          "G": 0.0070845012091283405,
          "R": 0.016574373828590967
        },
        "Avg_headroom_weighted": {
          "U": 0.9271792597577969,
          "G": 0.015678489177894113,
          "R": 0.057142251064309146
        },
        "Total_headroom": 40.85593333333331,
        "language_count": 6,
        "significance_filter": {
          "enabled": true,
          "alpha": 0.05
        }
      }
    },
```

For each dataset, the values under:

```
aggregate → Avg_headroom_weighted
```

correspond to the **weighted residual-based failure attribution shares** reported in **Figure 2** of the paper.

---

## 4. Translation Quality Evaluation (Figure 3)

To reproduce **Figure 3: Scatter plot of Reasoning Performance Ratio vs. Translation Quality**, translation quality must first be measured using **GEMBA-DA**.

Run the following script:

```bash
bash 4_measure_translation_ability.sh
```

This will generate a file of the form:

```
translation_eval_${MODEL_NAME}.json
```
Below is a sample of the output file.
```json
{
  "config": {
    "models": "Qwen/Qwen3-4B",
    "language_pairs": [
      "de-en",
      "es-en",
      "ar-en",
      "ja-en",
      "ko-en",
      "th-en",
      "bn-en",
      "sw-en",
      "te-en"
    ],
    "flores_split": "devtest",
    "max_samples": 100,
    "translation_temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 32768,
    "judge_model": "gpt-4.1"
  },
  "results": {
    "de-en": {
      "metrics": {
        "mean": 95.02,
        "stdev": 8.981068978690677,
        "min": 40.0,
        "max": 100.0
      },
```

For each language pair (e.g., `de-en`), the value:

```
metrics → mean
```

represents the **GEMBA-DA translation quality score**.

Before running `5_plot_correlation.sh`, please update the corresponding Python file with these measured translation quality values and Performance ratios.  
After doing so, running the script will reproduce **Figure 3** in the paper.
