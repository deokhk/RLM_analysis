# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?

![Paper Banner](assets/overview.png)

This is the official repository for the paper:

**Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?**  
Deokhyung Kang, Seonjeong Hwang, Daehui Kim, Hyounghun Kim, Gary Geunbae Lee  

📄 [[arXiv:2510.27269](https://arxiv.org/abs/2510.27269)]

---

## Overview

This repository provides the **analysis, detection, and mitigation framework** proposed in the paper:
*Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?*

The repository is designed as a **paper-aligned companion repository**, enabling direct and faithful reproduction of the paper’s main experimental results.

Specifically, this repo supports:
- **Residual-based attribution analysis** of multilingual reasoning gaps (Section 3)
- **Detection of understanding failures** from reasoning traces (Section 4)
- **Mitigation via Selective Translation** guided by failure detection (Section 5)

Each major section of the paper is mapped to a corresponding directory with a dedicated README explaining how to reproduce the results step by step.


---

## Requirements & Setup

### Environment Setup

All experiments can be run using **Conda** with Python 3.12.

```bash
conda create -n rlm_analysis python=3.12
conda activate rlm_analysis

pip install -e .
```

---

### Required External Resources

We use **FastText** for language identification.

Please download the FastText language identification model (`lid.176.ftz`) and move it to the `misc/` directory:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
mv lid.176.ftz ./misc/
```

---

## Paper ↔ Code Alignment

The repository structure is organized to closely mirror the paper’s experimental pipeline.

```
RLM_analysis/
├── experiments/              # Guide for reproducing the paper is provided here
├── misc/                     # Where the lid.176.ftz file should be placed
├── outputs/                  # Where outputs for all experiments will be saved
├── src/                      # Source code
├── translated_data/          # Translated queries
└── pyproject.toml            # For installing requirements
```

Each major component under the ./experiments directory corresponds to a section of the paper:

| Paper Section | Topic | Code Entry Point |
|---------------|-------|------------------|
| §3 | Why Does the Multilingual Reasoning Gap Emerge? | `3_multilingual_reasoning_gap_attribution/` |
| §4 | Detecting Understanding Failures | `4_detecting_understanding_failures/` |
| §5 | Selective Translation | `5_selective_translation/` |

Each directory contains its own **README** with detailed instructions for reproducing the corresponding experiments, including required inputs, scripts, and expected outputs.

---

## Pretrained Models and Released Outputs (Hugging Face)

To facilitate reproducibility, we provide pretrained models and experiment outputs via **Hugging Face Hub**.

---

### 1. Experiment Outputs and Prober Checkpoints  

We release experiment outputs for **Qwen3-4B**, including **prober checkpoints** and intermediate results, as a Hugging Face <img src="assets/hf_image.png" alt="Hugging Face" width="18" style="vertical-align: middle;"/> dataset: [**deokhk/multilingual_reasoning_gap_outputs**](https://huggingface.co/datasets/deokhk/multilingual_reasoning_gap_outputs)

You may download and place these files directly under:

```
./outputs/
```

You can download the files using the following example:

```python
from huggingface_hub import snapshot_download

repo_id = "deokhk/multilingual_reasoning_gap_outputs"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="./outputs/",
    local_dir_use_symlinks=False,
)
```

---

### 2. mmBERT Understanding Failure Detector  

The fine-tuned **mmBERT understanding failure detector** used in **Section 4** is released at the Hugging Face <img src="assets/hf_image.png" alt="Hugging Face" width="18" style="vertical-align: middle;"/> repository: [**deokhk/mmbert_ft_understandability_Qwen3-4B**](https://huggingface.co/deokhk/mmbert_ft_understandability_Qwen3-4B)


After downloading, place the contents under:

```
./outputs/experiments/mmbert_ft_understandability/Qwen3-4B/
```

---

## Reproducing the Paper

To reproduce the full experimental pipeline, we recommend running the sections **in order**:

1. **Section 3** – Multilingual reasoning gap attribution  
2. **Section 4** – Understanding failure detection  
3. **Section 5** – Selective Translation with failure-aware routing  

Please refer to the README inside each section directory for detailed, step-by-step instructions.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{kang2025multilingualreasoninggapsemerge,
      title={Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?}, 
      author={Deokhyung Kang and Seonjeong Hwang and Daehui Kim and Hyounghun Kim and Gary Geunbae Lee},
      year={2025},
      eprint={2510.27269},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.27269}, 
}
```

---

## License

This project is released under the Apache 2.0 License.
Please refer to individual datasets and model licenses for their respective terms.
