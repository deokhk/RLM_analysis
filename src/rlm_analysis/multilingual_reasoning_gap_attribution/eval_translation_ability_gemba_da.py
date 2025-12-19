#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate translation ability for multiple models on Flores-200 using GEMBA-DA.

- Generate translations with vLLM
- Score translations with GEMBA-DA (OpenAI judge)
- Supports:
  * facebook/flores (config="all", split=dev/devtest)
  * openlanguagedata/flores_plus (config per language, joined by id)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------------
# Parsing helper (reasoning vs final)
# -----------------------------
def divide_reasoning_trace_from_solution(pred: str, model_name: str = "Qwen3") -> Tuple[str, str]:
    if "gpt-oss" in model_name:
        pred = pred.replace("<|channel|>analysis<|message|>", "").replace("<|end|>", "").replace("<|return|>", "")
        splitted = pred.split("<|start|>assistant<|channel|>final<|message|>")
        if len(splitted) >= 2:
            reasoning_trace = splitted[0]
            solution = splitted[1]
            return (reasoning_trace, solution)
        else:
            logging.warning("Trace parsing failed for gpt-oss format; using full output as translation.")
            return (pred, pred)

    pred_splitted = pred.split("</think>")
    if len(pred_splitted) >= 2:
        reasoning_trace = pred_splitted[0]
        solution = pred_splitted[1]
        return (reasoning_trace, solution)

    logging.warning("No closing </think> found; using full output as translation.")
    return (pred, pred)


# -----------------------------
# Language settings
# -----------------------------
DEFAULT_LANGUAGE_PAIRS = (
    "de-en",
    "es-en",
    "ar-en",
    "ja-en",
    "ko-en",
    "th-en",
    "bn-en",
    "sw-en",
    "te-en",
)

FLORES_LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "bn": "ben_Beng",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "sw": "swh_Latn",
    "te": "tel_Telu",
    "th": "tha_Thai",
}

LANGUAGE_NAME_OVERRIDES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
}

DEFAULT_PROMPT_TEMPLATE = (
    "You are a professional translator. Translate the following {source_lang_name} text into {target_lang_name}. "
    "Preserve meaning, tone, and named entities. Provide only the translated text without any additional explanation.\n"
    "{source_lang_name} text: {source_text}\n\n{target_lang_name} translation:"
)


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Sample:
    sentence_id: str
    source_text: str
    reference_text: str


@dataclass
class TranslationResult(Sample):
    translation: str
    gemba_score: Optional[float]


# -----------------------------
# Language pair parsing
# -----------------------------
def parse_language_pairs(values: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in values:
        if "-" not in raw:
            raise ValueError(f"Language pair '{raw}' is missing '-' separator")
        src, tgt = raw.split("-", 1)
        pairs.append((src.strip(), tgt.strip()))
    return pairs


# -----------------------------
# GEMBA parsing (robust)
# -----------------------------
def parse_numerical_answer(answer: str, min: Optional[int] = None, max: Optional[int] = None):
    numbers = re.findall(r"\d+", answer)
    if len(numbers) == 1:
        return int(numbers[0])

    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def parse_and_check_numerical_answer(answer: str, min: int = 0, max: int = 100):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt
    return None


def validate_number(x: str, min: int = 0, max: int = 100):
    return parse_and_check_numerical_answer(x, min=min, max=max)


GEMBA_DA_PROMPT = (
    'Score the following translation from {source_lang} to {target_lang} '
    'on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" '
    'and a score of one hundred means "perfect meaning and grammar".\n\n'
    '{source_lang} source: "{source_seg}"\n'
    '{target_lang} translation: "{target_seg}"\n\n'
    'Return the result in strict JSON format with only one field "score".\n'
)

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_gemba_da_output(output_text: str) -> Optional[float]:
    if not isinstance(output_text, str):
        raise ValueError("Model output is not a string.")

    cleaned = output_text.strip()

    m = FENCE_RE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None
        score_val = parsed.get("score", None)
        if score_val is None:
            score_val = parsed.get("Score", None)
        if score_val is None:
            return None
        return float(score_val)
    except json.JSONDecodeError:
        v = validate_number(cleaned, min=0, max=100)
        return float(v) if v is not None else None
    except Exception:
        return None


# -----------------------------
# Dataset loading
# -----------------------------
def load_flores_pair_dataset(split: str, src_code: str, tgt_code: str) -> Dataset:
    """
    Returns a dataset with columns:
      - id
      - source_text
      - reference_text

    Supports:
      1) facebook/flores (config="all") with sentence_{langcode} columns
      2) openlanguagedata/flores_plus where each language is a config and has a single 'sentence' column
         -> join by 'id'
    """
    # 1) Try facebook/flores (all languages in one table)
    try:
        ds = load_dataset("facebook/flores", "all", split=split)
        return Dataset.from_dict(
            {
                "id": ds["id"],
                "source_text": ds[f"sentence_{src_code}"],
                "reference_text": ds[f"sentence_{tgt_code}"],
            }
        )
    except Exception as e:
        logging.warning(f"facebook/flores load failed for split={split}. Fallback to flores_plus. Reason: {e}")

    # 2) Fallback: flores_plus (per-language configs)
    ds_src = load_dataset("openlanguagedata/flores_plus", src_code, split=split)
    ds_tgt = load_dataset("openlanguagedata/flores_plus", tgt_code, split=split)

    # Typical columns: id, sentence
    src_map = {ex["id"]: ex["text"] for ex in ds_src}
    tgt_map = {ex["id"]: ex["text"] for ex in ds_tgt}

    ids = [ex["id"] for ex in ds_src if ex["id"] in tgt_map]
    if not ids:
        raise RuntimeError(f"flores_plus join produced 0 samples for {src_code}->{tgt_code} split={split}")

    return Dataset.from_dict(
        {
            "id": ids,
            "source_text": [src_map[i] for i in ids],
            "reference_text": [tgt_map[i] for i in ids],
        }
    )


def build_samples_from_pair_dataset(pair_ds: Dataset, max_samples: Optional[int]) -> List[Sample]:
    selected: List[Sample] = []
    iterable = pair_ds if max_samples is None else pair_ds.select(range(min(len(pair_ds), max_samples)))
    for item in iterable:
        selected.append(
            Sample(
                sentence_id=item["id"],
                source_text=item["source_text"],
                reference_text=item["reference_text"],
            )
        )
    return selected


# -----------------------------
# Prompt building & generation
# -----------------------------
def build_translation_prompts(samples: Sequence[Sample], src_name: str, tgt_name: str, template: str):
    prompts = []
    for sample in samples:
        prompts.append(
            template.format(
                source_lang_name=src_name,
                target_lang_name=tgt_name,
                source_text=sample.source_text,
            )
        )
    prompts_dict = [[{"role": "user", "content": p}] for p in prompts]
    return prompts_dict


def generate_translations(
    llm: LLM,
    samples: Sequence[Sample],
    model_id: str,
    tokenizer,
    src_name: str,
    tgt_name: str,
    template: str,
    sampling_params: SamplingParams,
) -> List[TranslationResult]:
    results: List[TranslationResult] = []
    prompts_dict = build_translation_prompts(samples, src_name, tgt_name, template)
    formatted_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts_dict
    ]

    logging.info(f"Generating {len(formatted_prompts)} translations with {model_id}, {src_name} -> {tgt_name}")
    generations = llm.generate(formatted_prompts, sampling_params=sampling_params)

    for sample, generation in zip(samples, generations):
        if "gpt-oss" in model_id:
            pred_res = tokenizer.decode(generation.outputs[0].token_ids).strip()
        else:
            pred_res = generation.outputs[0].text.strip()

        (_, prediction) = divide_reasoning_trace_from_solution(pred_res, model_id)
        results.append(
            TranslationResult(
                sentence_id=sample.sentence_id,
                source_text=sample.source_text,
                reference_text=sample.reference_text,
                translation=prediction.strip(),
                gemba_score=None,
            )
        )
    return results


# -----------------------------
# GEMBA scoring (async)
# -----------------------------
async def run_gemba_da_async(client: AsyncOpenAI, prompt: str, model_name: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0,
        )
        return chat_completion.choices[0].message.content


async def run_gemba_das(
    client: AsyncOpenAI,
    prompts: List[str],
    model_name: str,
    parse_fn,
    max_concurrent_requests: int = 30,
) -> List[Optional[float]]:
    semaphore = asyncio.Semaphore(value=max_concurrent_requests)
    tasks = [run_gemba_da_async(client, prompt, model_name, semaphore) for prompt in prompts]
    outputs = await asyncio.gather(*tasks)
    return [parse_fn(res) for res in outputs]


def score_with_gemba_da(
    judge_client: AsyncOpenAI,
    judge_model_name: str,
    max_concurrent_requests: int,
    src_lang: str,
    tgt_lang: str,
    translations: List[TranslationResult],
) -> None:
    prompt_list: List[str] = []
    for result in translations:
        payload = {
            "source_lang": LANGUAGE_NAME_OVERRIDES[src_lang],
            "target_lang": LANGUAGE_NAME_OVERRIDES[tgt_lang],
            "source_seg": result.source_text,
            "target_seg": result.translation,
        }
        prompt_list.append(GEMBA_DA_PROMPT.format(**payload))

    logging.info(f"Scoring {src_lang}->{tgt_lang} with judge={judge_model_name} (n={len(prompt_list)})")
    scores = asyncio.run(
        run_gemba_das(
            judge_client,
            prompt_list,
            model_name=judge_model_name,
            parse_fn=parse_gemba_da_output,
            max_concurrent_requests=max_concurrent_requests,
        )
    )
    for score, result in zip(scores, translations):
        result.gemba_score = float(score) if score is not None else None


# -----------------------------
# Aggregation & serialization
# -----------------------------
def aggregate_scores(translations: Sequence[TranslationResult]) -> Dict[str, Optional[float]]:
    numeric_scores = [t.gemba_score for t in translations if t.gemba_score is not None]
    if not numeric_scores:
        return {"mean": None, "stdev": None, "min": None, "max": None}
    if len(numeric_scores) == 1:
        v = numeric_scores[0]
        return {"mean": v, "stdev": 0.0, "min": v, "max": v}
    return {
        "mean": mean(numeric_scores),
        "stdev": pstdev(numeric_scores),
        "min": min(numeric_scores),
        "max": max(numeric_scores),
    }


def to_serializable(results_by_pair: Dict[str, List[TranslationResult]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for pair, translations in results_by_pair.items():
        metrics = aggregate_scores(translations)
        out[pair] = {
            "metrics": metrics,
            "num_samples": len(translations),
            "scored_samples": sum(1 for t in translations if t.gemba_score is not None),
            "segment_scores": [
                {
                    "sentence_id": t.sentence_id,
                    "source": t.source_text,
                    "reference": t.reference_text,
                    "translation": t.translation,
                    "gemba_score": t.gemba_score,
                }
                for t in translations
            ],
        }
    return out


# -----------------------------
# Evaluation core
# -----------------------------
def run_evaluation(args: argparse.Namespace) -> Dict[str, object]:
    language_pairs = parse_language_pairs(args.language_pairs)
    language_pair_labels = [f"{src}-{tgt}" for src, tgt in language_pairs]

    api_key = open(args.openai_key_path).read().strip() if args.openai_key_path else None
    judge_client = AsyncOpenAI(api_key=api_key)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    model_id = args.model_name
    logging.info("Evaluating model: %s", model_id)

    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 40960 if "phi-4" not in model_id.lower() else 32768,
    }
    llm = LLM(**llm_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    results_by_pair: Dict[str, List[TranslationResult]] = {}

    for src, tgt in language_pairs:
        if src not in FLORES_LANGUAGE_CODES or tgt not in FLORES_LANGUAGE_CODES:
            raise KeyError(f"Unsupported language pair: {src}-{tgt}")
        if src not in LANGUAGE_NAME_OVERRIDES or tgt not in LANGUAGE_NAME_OVERRIDES:
            raise KeyError(f"Missing language name mapping for: {src}-{tgt}")

        src_code = FLORES_LANGUAGE_CODES[src]
        tgt_code = FLORES_LANGUAGE_CODES[tgt]
        src_name = LANGUAGE_NAME_OVERRIDES[src]
        tgt_name = LANGUAGE_NAME_OVERRIDES[tgt]

        # Pair-specific loading (supports flores_plus)
        pair_ds = load_flores_pair_dataset(args.flores_split, src_code, tgt_code)
        samples = build_samples_from_pair_dataset(pair_ds, args.max_samples)

        translations = generate_translations(
            llm=llm,
            samples=samples,
            model_id=model_id,
            tokenizer=tokenizer,
            src_name=src_name,
            tgt_name=tgt_name,
            template=args.prompt_template,
            sampling_params=sampling_params,
        )

        score_with_gemba_da(
            judge_client=judge_client,
            judge_model_name=args.judge_model,
            max_concurrent_requests=args.max_concurrent_requests,
            src_lang=src,
            tgt_lang=tgt,
            translations=translations,
        )

        pair_label = f"{src}-{tgt}"
        results_by_pair[pair_label] = translations

    aggregated_results = to_serializable(results_by_pair)

    # vLLM object cleanup
    del llm

    return {
        "config": {
            "model": args.model_name,
            "language_pairs": language_pair_labels,
            "flores_split": args.flores_split,
            "max_samples": args.max_samples,
            "translation_temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "judge_model": args.judge_model,
            "max_concurrent_requests": args.max_concurrent_requests,
        },
        "results": aggregated_results,
    }


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Flores-200 translations with GEMBA-DA.")
    parser.add_argument("--model_name", required=True, help="Model name to load with vLLM")
    parser.add_argument("--output_dir", required=True, help="Directory for storing evaluation results JSON")
    parser.add_argument("--language_pairs", nargs="+", default=DEFAULT_LANGUAGE_PAIRS, help="Pairs like de-en")
    parser.add_argument("--flores_split", default="devtest", help="Split to evaluate (dev or devtest)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per language pair")

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32768)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)

    parser.add_argument("--judge_model", default="gpt-4.1", help="OpenAI model for GEMBA-DA scoring")
    parser.add_argument("--openai_key_path", required=True, help="Path to OpenAI API key file")
    parser.add_argument("--max_concurrent_requests", type=int, default=30)

    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results = run_evaluation(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"translation_eval_{args.model_name.replace('/', '_')}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    logging.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
