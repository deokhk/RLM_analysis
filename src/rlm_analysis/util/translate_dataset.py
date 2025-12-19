import json
import random
import argparse
from datasets import load_dataset
import os
import re 
import asyncio
from openai import AsyncOpenAI
import logging
from tqdm.asyncio import tqdm
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open("/home/deokhk/research/ZX-seq2seq/nlplab2_openai_key.txt", "r") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

# Initialize the OpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=3,
)

# "mmlu-prox-lite": ["de", "es", "ar", "ja", "ko", "th", "bn", "sw", "te"]

dataset_to_source_languages = {
    "polymath": ["de", "es", "ar", "ja", "ko", "th", "bn", "sw", "te"],
    "mmlu-prox-lite": ["de", "es", "ar", "ja", "ko", "th", "bn", "sw", "te"]
}

translate_to_xx_code_to_language = {
    "fr": "French",
    "mr": "Marathi",
    "wo": "Wolof"
}


async def translate_to_xx_from_english_instruction_async_polymath(instruction, language, model_name, semaphore):
    prompt = f"""Translate the following mathematical question enclosed within <instruction> and </instruction> into {language}.  
The text may contain mathematical notation and LaTeX formatting. You must strictly preserve:  
- All LaTeX math and commands EXACTLY as written, including inline math ($...$), display math ($$...$$), \(...\), \[...\], and any \begin{...}...\end{...} environments.  
- All mathematical symbols, variables, numbers, operators, and equation labels.  

Provide only the translated instruction without any additional explanation.  
Wrap the translated output with <translated> and </translated> tags.  
""" + f"<instruction>{instruction}</instruction>\n"

    async with semaphore:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0
        )
        translated_text = chat_completion.choices[0].message.content
        return translated_text



async def translate_instruction_async_polymath(instruction, model_name, semaphore):
    prompt = """Translate the following mathematical question enclosed within <instruction> and </instruction> into English.  
The text may contain mathematical notation and LaTeX formatting. You must strictly preserve:  
- All LaTeX math and commands EXACTLY as written, including inline math ($...$), display math ($$...$$), \(...\), \[...\], and any \begin{...}...\end{...} environments.  
- All mathematical symbols, variables, numbers, operators, and equation labels.  

Provide only the translated instruction without any additional explanation.  
Wrap the translated output with <translated> and </translated> tags.  
""" + f"<instruction>{instruction}</instruction>\n"

    async with semaphore:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0
        )
        translated_text = chat_completion.choices[0].message.content
        return translated_text
    

async def translate_instruction_and_option_async_mmlu_prox_lite(datapoint, model_name, semaphore):

    input_json = {}
    
    # Always include question
    if "question" in datapoint:
        input_json["question"] = datapoint["question"]

    # Extract options dynamically (0~9, but may not all exist)
    for i in range(10):
        key = f"option_{i}"
        if key in datapoint:
            input_json[key] = datapoint[key]

    payload = json.dumps(input_json, ensure_ascii=False)
    prompt = f"""
    You are a professional scientific translator.

    TASK
    - Translate ONLY the string **values** in the given JSON object into English.
    - Do not change the JSON keys, structure, or order.
    - Preserve all numbers, mathematical expressions, symbols, and units exactly.
    - Return ONLY the translated JSON object.

    INPUT:
    {payload}
    """

    async with semaphore:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0
        )
        translated_output = chat_completion.choices[0].message.content
        return translated_output

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')

def parse_translation_output(output_text: str) -> Dict[str, Any]:
    """
    Robustly parse a JSON object possibly wrapped in ```json ... ``` fences.
    Also fixes invalid backslash escapes common in LaTeX (e.g., \frac → \\frac).
    If still invalid, return a dict with raw output for manual inspection.
    """
    if not isinstance(output_text, str):
        raise ValueError("Model output is not a string.")

    cleaned = output_text.strip()

    # 1) Remove triple backtick fences if present
    m = FENCE_RE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    # 2) First attempt: direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed object is not a dictionary.")
        return parsed
    except json.JSONDecodeError:
        pass

    # 3) Second attempt: fix invalid backslash escapes (e.g., \frac → \\frac)
    fixed = INVALID_ESCAPE_RE.sub(r'\\\\', cleaned)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed object is not a dictionary.")
        return parsed
    except json.JSONDecodeError:
        pass

    # 4) Final fallback: return raw output for manual inspection
    return {"__raw_output__": output_text}


async def translate_questions_polymath(datapoints, model_name):
    semaphore = asyncio.Semaphore(value=30)

    questions_t_tasks = [translate_instruction_async_polymath(datapoint['question'], model_name, semaphore) for datapoint in datapoints]

    translated_questions = await tqdm.gather(*questions_t_tasks)

    translated_questions_post_processed = [x.replace("<translated>", "").replace("</translated>", "").strip() for x in translated_questions]
    return translated_questions_post_processed

async def translate_questions_to_xx_polymath(datapoints, target_language, model_name):
    semaphore = asyncio.Semaphore(value=30)

    questions_t_tasks = [translate_to_xx_from_english_instruction_async_polymath(datapoint['question'], target_language, model_name, semaphore) for datapoint in datapoints]
    translated_questions = await tqdm.gather(*questions_t_tasks)

    translated_questions_post_processed = [x.replace("<translated>", "").replace("</translated>", "").strip() for x in translated_questions]
    return translated_questions_post_processed


async def translate_datapoints_mmlu_prox_lite(datapoints, model_name):
    semaphore = asyncio.Semaphore(value=30)

    t_tasks = [translate_instruction_and_option_async_mmlu_prox_lite(datapoint, model_name, semaphore) for datapoint in datapoints]

    translated_outputs = await tqdm.gather(*t_tasks)

    translated_outputs_dict = [parse_translation_output(x) for x in translated_outputs]
    return translated_outputs_dict


def collect_options_from_datapoint(datapoint: Dict[str, Any]) -> List[str]:
    """
    Collect valid options from option_0 .. option_9.
    Exclude None and 'N/A'.
    """
    options: List[str] = []
    for i in range(0, 10):  # option_0 .. option_9
        key = f"option_{i}"
        if key in datapoint and datapoint[key] is not None and datapoint[key] != "N/A":
            options.append(datapoint[key])
    return options

def format_options_string(options: List[str]) -> str:
    """
    Format a multiple-choice options string as:
    (A) opt1
    (B) opt2
    ...
    """
    letters = ["A","B","C","D","E","F","G","H","I","J"]
    out = ""
    for idx, opt in enumerate(options):
        out += f"({letters[idx]}) {opt}\n"
    out = out.rstrip("\n")  # Remove trailing newline
    return out



async def main(dataset_name, model_name, save_base_dir, translate_to_xx=False, translate_to_xx_target_languages=['fr', 'mr', 'wo']):

    # Load the dataset
    if dataset_name == "polymath":
        if translate_to_xx:
            splits = ["low"]
        else:
            splits = ["low", "medium", "high"]
        save_dir = os.path.join(save_base_dir, "polymath")
        # 바1: 스플릿 진행률
        for split in tqdm(splits, desc="Processing splits", position=0):
            dataset_languages = dataset_to_source_languages[dataset_name]
            if translate_to_xx:
                dataset_languages = translate_to_xx_target_languages
            translated_dataset_dict = {}

            # 바2: 언어 진행률 (현재 언어 표시)
            lang_bar = tqdm(dataset_languages,
                            desc=f"Languages | split={split}",
                            position=1, leave=False, dynamic_ncols=True)

            for lang in lang_bar:
                lang_bar.set_postfix_str(f"lang={lang}")  # 👈 현재 처리 중인 언어 표시

                if lang not in translated_dataset_dict:
                    translated_dataset_dict[lang] = []

                # tqdm 출력 흐트러짐을 줄이려면 logging 대신 tqdm.write 사용 권장

                language_dataset = load_dataset("Qwen/PolyMath", lang, split=split) if not translate_to_xx else load_dataset("Qwen/PolyMath", "en", split=split)
                if translate_to_xx:
                    tqdm.write(f"[INFO] Translating English PolyMath → {translate_to_xx_code_to_language[lang]} ({split})")
                    # 비동기 번역 호출. Here, the language_dataset is English.
                    translated_language_questions = await translate_questions_to_xx_polymath(
                        language_dataset, translate_to_xx_code_to_language[lang], model_name
                    )
                else:
                    # 비동기 번역 호출
                    tqdm.write(f"[INFO] Translating {lang} PolyMath → English ({split})")
                    translated_language_questions = await translate_questions_polymath(
                        language_dataset, model_name
                    )

                for datapoint, translated_question in zip(language_dataset, translated_language_questions):
                    datapoint["translated_question"] = translated_question
                    translated_dataset_dict[lang].append(datapoint)

            # 언어 바 정리
            lang_bar.close()

            os.makedirs(save_dir, exist_ok=True)
            if translate_to_xx:
                out_path = os.path.join(save_dir, f"polymath_{split}_translated_to_{"_".join(translate_to_xx_target_languages)}.json")
            else:
                out_path = os.path.join(save_dir, f"polymath_{split}_translated_to_english.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(translated_dataset_dict, f, ensure_ascii=False, indent=4)

            tqdm.write(f"[INFO] Finished Polymath {split}. Saved to {out_path}")
    
    elif dataset_name == "mmlu-prox-lite":
        save_dir = os.path.join(save_base_dir, "mmlu-prox-lite")
        os.makedirs(save_dir, exist_ok=True)

        dataset_languages = dataset_to_source_languages[dataset_name]
        translated_dataset_dict = {}
        reasoning_categories = ["math", "physics", "chemistry", "computer science", "engineering"] 

        manual_review_count =0
        # 바2: 언어 진행률 (현재 언어 표시)
        lang_bar = tqdm(dataset_languages,
                        desc=f"Languages | split=test",
                        position=1, leave=False, dynamic_ncols=True)
        for lang in lang_bar:
            lang_bar.set_postfix_str(f"lang={lang}")  # 👈 현재 처리 중인 언어 표시

            if lang not in translated_dataset_dict:
                translated_dataset_dict[lang] = []

            # tqdm 출력 흐트러짐을 줄이려면 logging 대신 tqdm.write 사용 권장
            tqdm.write(f"[INFO] Translating {lang} mmlu-prox-lite → English (test)")

            language_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="test")
            reasoning_subset_language_dataset = language_dataset.filter(lambda x: x["category"] in reasoning_categories)

            # 비동기 번역 호출
            translated_dicts = await translate_datapoints_mmlu_prox_lite(
                reasoning_subset_language_dataset, model_name
            )

            for datapoint, translated_dict in zip(reasoning_subset_language_dataset, translated_dicts):

                if "__raw_output__" in translated_dict:
                    tqdm.write(f"[WARNING] Failed to parse translation for datapoint id={datapoint.get('id', 'N/A')}. Storing raw output.")
                    datapoint["translated_output"] = translated_dict["__raw_output__"]
                    datapoint["requires_manual_review"] = True
                    translated_dataset_dict[lang].append(datapoint)
                    manual_review_count +=1
                    continue  # Skip further processing for this datapoint

                options_list = collect_options_from_datapoint(translated_dict)
                option_string = format_options_string(options_list)

                datapoint["translated_question"] = translated_dict["question"]
                datapoint["translated_options_string"] = option_string
                translated_dataset_dict[lang].append(datapoint)
        # 언어 바 정리
        lang_bar.close()

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"mmlu_prox_lite_test_translated_to_english.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated_dataset_dict, f, ensure_ascii=False, indent=4)

        tqdm.write(f"[INFO] Finished mmlu-prox-lite test. Saved to {out_path}")
        tqdm.write(f"[INFO] Total datapoints requiring manual review: {manual_review_count}")
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate instructions using OpenAI API")
    parser.add_argument("--dataset_name", type=str, default="polymath", choices=["polymath", "mmlu-prox-lite"], help="Dataset name to use for translation")
    parser.add_argument("--model_name", type=str, default="gpt-4.1", help="Model name to use for translation")
    parser.add_argument("--save_base_dir", type=str, default="/home/deokhk/research/LRM_analysis/translated_data/", help="Directory to save the translated data")

    parser.add_argument("--translate_to_xx", action="store_true", help="If set, translate from English to other languages")
    parser.add_argument("--translate_to_xx_target_languages", type=str, nargs="+", default=['fr', 'mr', 'wo'], help="Target languages for translation to xx")
    args = parser.parse_args()

    asyncio.run(main(args.dataset_name, args.model_name, args.save_base_dir, args.translate_to_xx, args.translate_to_xx_target_languages))