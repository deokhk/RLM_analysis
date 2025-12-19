"""
MGSM: Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems. 
Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei
https://arxiv.org/abs/2210.03057 reference: https://github.com/google-research/url-nlp 
"""

import re
import os
from tqdm import tqdm
from rlm_analysis.eval.scripts import math_equal
from typing import Optional, Union
from math_verify import parse, verify

os.environ["TOKENIZERS_PARALLELISM"] = "false"



MMLU_PROX_LITE_LANG_TO_ANSWER_REGEX = {
    "en": r"answer is \(?([ABCDEFGHIJ])\)?",
    "de": r"Die Antwort ist \(?([ABCDEFGHIJ])\)?", 
    "es": r"La respuesta es \(?([ABCDEFGHIJ])\)?",
    "ar": r"الإجابة هي \(?([ABCDEFGHIJ])\)?", 
    "ja": r"答えは \(?([ABCDEFGHIJ])\)? です", 
    "ko": r"답은 \(?([ABCDEFGHIJ])\)?입니다",
    "th": r"คำตอบคือ \(?([ABCDEFGHIJ])\)?",
    "bn": r"উত্তর হল \(?([ABCDEFGHIJ])\)?",
    "sw": r"Jibu ni \(?([ABCDEFGHIJ])\)?",
    "te": r"సమాధానం \(?([ABCDEFGHIJ])\)?",
    "pt": r"A resposta é \(?([ABCDEFGHIJ])\)?",
    "it": r"La risposta è \(?([ABCDEFGHIJ])\)?",
    "fr": r"La réponse est \(?([ABCDEFGHIJ])\)?",
    "id": r"Jawabannya adalah \(?([ABCDEFGHIJ])\)?",
    "vi": r"Câu trả lời là \(?([ABCDEFGHIJ])\)?",
    "wo": r"Tontu bi mooy \(?([ABCDEFGHIJ])\)?",
    "mr": r"उत्तर आहे \(?([ABCDEFGHIJ])\)?",
    "yo": r"Ìdáhùn náà ni \(?([ABCDEFGHIJ])\)?"
}

# For MGSM

def parse_answer(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        return ""

    answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the last number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""

def parse_answer_english_reasoning(answer: str) -> str:
    # Remove the answer prefix
    answer_text = answer.split(":")[-1].strip()
    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


# For polymath (taken from https://github.com/QwenLM/PolyMath/blob/fbf4e41cae78687d6be7447dbea897357c06aaa7/eval/scripts.py)
_BOXED_OPEN_RE = re.compile(r'\\{1,2}boxed\s*\{', re.IGNORECASE)

def _find_matching_rbrace(text: str, start_idx: int) -> int:
    r"""
    Given the index right after an opening '{', find the index of the matching '}'.
    - Supports nested braces.
    - Skips escaped chars like '\{' and '\}' so they don't affect depth.
    Returns the index of the matching '}' or -1 if not found.
    """
    depth = 1
    i = start_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '\\':        # skip escaped char
            i += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def extract_last_boxed(text: str, return_full: bool = True) -> str:
    """
    Return the last balanced \boxed{...} (or \\boxed{...}) occurrence.
    - return_full=True  -> return the full block, e.g., "\\boxed{FINAL_ANSWER}"
    - return_full=False -> return only the inner content, e.g., "FINAL_ANSWER"
    Returns "" if none found or if braces are unbalanced.
    """
    if not text:
        return ""
    matches = list(_BOXED_OPEN_RE.finditer(text))
    for m in reversed(matches):
        start_content = m.end()                # right after '{'
        end = _find_matching_rbrace(text, start_content)
        if end != -1:
            return text[m.start():end+1] if return_full else text[start_content:end]
    return ""


def extract_last_option(text: str, lang) -> str:
    pattern = MMLU_PROX_LITE_LANG_TO_ANSWER_REGEX[lang]
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]   # Return final match
    else:
        eng_pattern = MMLU_PROX_LITE_LANG_TO_ANSWER_REGEX["en"]
        eng_matches = re.findall(eng_pattern, text)
        if eng_matches:
            return eng_matches[-1]
        return None


from rlm_analysis.util.parser_llm import extract_answers_from_text_list, extract_option_answers_from_text_list


def extract_boxed_or_from_llm(parser_llm, parser_tokenizer, samples, pred_results, pred_target_key="prediction", last_n_tokens=512):
    # First extract "boxed" answers 
    extracted_answer_dict = {}

    for sample, pred_result in zip(samples, pred_results):
        assert sample["id"] == pred_result["id"]
        extracted_answer = extract_last_boxed(pred_result[pred_target_key])
        if extracted_answer != "": 
            extracted_answer_dict[sample["id"]] = extracted_answer

    target_boxed_answer_not_found_dicts = {}

    for sample, pred_result in zip(samples, pred_results):
        sid = sample["id"]
        if sid not in extracted_answer_dict:
            target_boxed_answer_not_found_dicts[sid] = {
                "question": sample["question"],
                "text": pred_result[pred_target_key]
            }
    print(f"Number of samples where boxed answer not found in {pred_target_key}: "
        f"{len(target_boxed_answer_not_found_dicts)} / {len(samples)}")

    # ---------- LLM-based parsing ----------
    if len(target_boxed_answer_not_found_dicts) > 0:
        # Preseve the key in fixed order
        _target_keys = list(target_boxed_answer_not_found_dicts.keys())
        _target_questions = [target_boxed_answer_not_found_dicts[k]["question"] for k in _target_keys]
        _target_texts     = [target_boxed_answer_not_found_dicts[k]["text"]     for k in _target_keys]

        try:
            pred_extracted_answers = extract_answers_from_text_list(
                parser_llm,
                parser_tokenizer,
                _target_questions,
                _target_texts,
                last_n_tokens=last_n_tokens
            )
        except Exception as e:
            print(f"Parser LLM failed on predictions: {e}")
            pred_extracted_answers = [None] * len(_target_keys)

        for k, extracted_answer in zip(_target_keys, pred_extracted_answers):
            if extracted_answer != "":
                extracted_answer_dict[k] = extracted_answer.strip()
            else:
                extracted_answer_dict[k] = None

        return (_target_keys, extracted_answer_dict)
    else:
        return ([], extracted_answer_dict)


def extract_from_option_expr_or_from_llm(parser_llm, parser_tokenizer, samples, pred_results, lang, pred_target_key="prediction", last_n_tokens=512):
    extracted_answer_dict = {}
    target_option_answer_not_found_dicts = {}
    for sample, pred_result in zip(samples, pred_results):
        sid = sample["id"]
        option_answer = extract_last_option(pred_result[pred_target_key], lang)
        if option_answer != None: 
            extracted_answer_dict[sample["id"]] = option_answer

        if sid not in extracted_answer_dict:
            # Construct options string 
            sample_option_dict = sample["options_dict"]
            options_string = ""
            for option_letter, option in sample_option_dict.items():
                options_string += f"({option_letter}) {option}, "
            # Remove last comma and space
            options_string = options_string.rstrip(", ")

            target_option_answer_not_found_dicts[sid] = {
                "options_string": options_string,
                "text": pred_result[pred_target_key]
            }
    print(f"Number of samples where option answer not found in {pred_target_key}: "
        f"{len(target_option_answer_not_found_dicts)} / {len(samples)}")

    # ---------- LLM-based parsing ----------
    if len(target_option_answer_not_found_dicts) > 0:
        # Preseve the key in fixed order
        _target_keys = list(target_option_answer_not_found_dicts.keys())
        _target_option_blocks_in_string_format_list = [target_option_answer_not_found_dicts[k]["options_string"] for k in _target_keys]
        _target_texts     = [target_option_answer_not_found_dicts[k]["text"]     for k in _target_keys]

        try:
            pred_extracted_answers = extract_option_answers_from_text_list(
                parser_llm,
                parser_tokenizer,
                _target_option_blocks_in_string_format_list,
                _target_texts,
                last_n_tokens=last_n_tokens
            )
        except Exception as e:
            print(f"Parser LLM failed on predictions: {e}")
            pred_extracted_answers = [None] * len(_target_keys)

        for k, extracted_answer in zip(_target_keys, pred_extracted_answers):
            if extracted_answer != "":
                extracted_answer_dict[k] = extracted_answer.strip()
            else:
                extracted_answer_dict[k] = None
        return (_target_keys, extracted_answer_dict)
    else:
        return ([], extracted_answer_dict)




class MGSMEvaluator():
    def __init__(
            self,
            language_code,
            samples,
            parser_llm,
            parser_tokenizer
    ):
        self.language_code = language_code
        self.samples = samples
        self.parser_llm = parser_llm
        self.parser_tokenizer = parser_tokenizer

    def evaluate(self, pred_results):
        eval_results = []

        print("Extracting answers from prediction and reasoning traces..")

        # (1) Extract answers from prediction and reasoning_trace
        #     If extraction fails, use parser LLM as fallback.
        _, pred_extracted_answer_dict = extract_boxed_or_from_llm(
            self.parser_llm,
            self.parser_tokenizer,
            self.samples,
            pred_results,
            pred_target_key="prediction",
            last_n_tokens=512
        )
        _, trace_extracted_answer_dict = extract_boxed_or_from_llm(
            self.parser_llm,
            self.parser_tokenizer,
            self.samples,
            pred_results,
            pred_target_key="reasoning_trace",
            last_n_tokens=512
        )

        for sample, pred_result in zip(self.samples, pred_results):
            assert sample["id"] == pred_result["id"]
            # For polymath, this "parse" using math-verify library is needed.
            extracted_answer = parse(pred_result["prediction"])
            if extracted_answer is not None: 
                pred_extracted_answer_dict[sample["id"]] = extracted_answer


        for sample, pred_result in zip(self.samples, pred_results):
            assert sample["id"] == pred_result["id"]
            extracted_answer = parse(pred_result["reasoning_trace"])
            if extracted_answer is not None: 
                trace_extracted_answer_dict[sample["id"]] = extracted_answer


        # (2) Evaluate predictions against gold answers
        for sample, pred_result in tqdm(zip(self.samples, pred_results),
                                        desc="Evaluating...",
                                        total=len(self.samples)):
            assert sample["id"] == pred_result["id"]

            sid = sample["id"]
            answer = parse(str(sample["answer"]))

            # Extracted answers
            pred_answer  = pred_extracted_answer_dict.get(sid, None)
            trace_answer = trace_extracted_answer_dict.get(sid, None)

            # Save extracted answers in pred_result
            pred_result["pred_answer"] = pred_answer
            pred_result["pred_answer_from_trace"] = trace_answer


            # Evaluate correctness
            score_from_pred = verify(answer, pred_answer)
            score_from_trace = verify(answer, trace_answer)

            # Save evaluation results back into pred_result
            pred_result["score"] = 1.0 if score_from_pred else 0.0
            pred_result["score_from_trace"] = 1.0 if score_from_trace else 0.0


            eval_results.append(
                {
                    "id": sample["id"],
                    "language_code": self.language_code,
                    "pred_answer": pred_answer,
                    "score": 1.0 if score_from_pred else 0.0,
                    "pred_answer_from_trace": trace_answer,
                    "score_from_trace": 1.0 if score_from_trace else 0.0,
                }
            )

        return eval_results


class PolyMathEvaluator():
    def __init__(
            self,
            language_code,
            samples,
            parser_llm,
            parser_tokenizer
    ):
        self.language_code = language_code
        self.samples = samples
        self.parser_llm = parser_llm
        self.parser_tokenizer = parser_tokenizer

    
    def evaluate(self, pred_results):
        eval_results = []
        print("Extracting answers from prediction and reasoning traces..")

        pred_boxed_not_found_ids, pred_extracted_answer_dict = extract_boxed_or_from_llm(self.parser_llm, self.parser_tokenizer, self.samples, pred_results, pred_target_key="prediction", last_n_tokens=512)
        trace_boxed_not_found_ids, trace_extracted_answer_dict = extract_boxed_or_from_llm(self.parser_llm, self.parser_tokenizer, self.samples, pred_results, pred_target_key="reasoning_trace", last_n_tokens=512)
        for sample, pred_result in zip(self.samples, pred_results):
            assert sample["id"] == pred_result["id"]
            # For polymath, this "parse" using math-verify library is needed.
            extracted_answer = parse(pred_result["prediction"])
            if extracted_answer is not None: 
                pred_extracted_answer_dict[sample["id"]] = extracted_answer


        for sample, pred_result in zip(self.samples, pred_results):
            assert sample["id"] == pred_result["id"]
            extracted_answer = parse(pred_result["reasoning_trace"])
            if extracted_answer is not None: 
                trace_extracted_answer_dict[sample["id"]] = extracted_answer


        for sample in tqdm(self.samples, desc="Evaluating...", total=len(self.samples)):
            sid = sample["id"]
            answer = parse(str(sample["answer"]))
            sample["pred_answer"] = pred_extracted_answer_dict.get(sid, None)
            sample["pred_answer_from_trace"] = trace_extracted_answer_dict.get(sid, None)

            pred_answer  = sample["pred_answer"]
            trace_answer = sample["pred_answer_from_trace"]


            # Evaluate correctness
            score_from_pred = verify(answer, pred_answer)
            score_from_trace = verify(answer, trace_answer)

            eval_results.append(
                {
                    "id": sample["id"],
                    "language_code": self.language_code,
                    "pred_answer": sample["pred_answer"],
                    "score": 1.0 if score_from_pred else 0.0,
                    "pred_answer_from_trace": sample["pred_answer_from_trace"],
                    "score_from_trace": 1.0 if score_from_trace else 0.0,
                    "both_extracted_from_regex": (sid not in pred_boxed_not_found_ids and sid not in trace_boxed_not_found_ids)
                }
            )


        return eval_results        
        

class MMLUProXLiteEvaluator():
    def __init__(
            self,
            language_code,
            samples,
            parser_llm,
            parser_tokenizer
    ):
        self.language_code = language_code
        self.samples = samples
        self.parser_llm = parser_llm
        self.parser_tokenizer = parser_tokenizer

    def is_two_option_same(self, pred_answer: Optional[str], gold_answer: str) -> bool:
        if pred_answer is None:
            return False
        pred_answer = pred_answer.strip().upper()
        gold_answer = gold_answer.strip().upper()
        return pred_answer == gold_answer
    
    def evaluate(self, pred_results):
        eval_results = []
        print("Extracting answers from prediction and reasoning traces..")

        _, pred_extracted_answer_dict = extract_from_option_expr_or_from_llm(self.parser_llm, self.parser_tokenizer, self.samples, pred_results, lang=self.language_code, pred_target_key="prediction", last_n_tokens=512)
        _, trace_extracted_answer_dict = extract_from_option_expr_or_from_llm(self.parser_llm, self.parser_tokenizer, self.samples, pred_results, lang=self.language_code, pred_target_key="reasoning_trace", last_n_tokens=512)

        for sample in tqdm(self.samples, desc="Evaluating...", total=len(self.samples)):
            sid = sample["id"]
            answer = sample["answer"]
            sample["pred_answer"] = pred_extracted_answer_dict.get(sid, None)
            sample["pred_answer_from_trace"] = trace_extracted_answer_dict.get(sid, None)

            pred_answer  = sample["pred_answer"]
            trace_answer = sample["pred_answer_from_trace"]


            # Evaluate correctness
            score_from_pred = self.is_two_option_same(pred_answer, answer)
            score_from_trace = self.is_two_option_same(trace_answer, answer)

            eval_results.append(
                {
                    "id": sample["id"],
                    "language_code": self.language_code,
                    "pred_answer": sample["pred_answer"],
                    "score": 1.0 if score_from_pred else 0.0,
                    "pred_answer_from_trace": sample["pred_answer_from_trace"],
                    "score_from_trace": 1.0 if score_from_trace else 0.0,
                }
            )

        return eval_results        
