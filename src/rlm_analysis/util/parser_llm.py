import re 
import random
from vllm import SamplingParams
from rlm_analysis.eval.evaluator import extract_last_boxed

def take_last_n_tokens(text: str, tokenizer, n: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    tail_ids = ids[-n:] if len(ids) > n else ids
    return tokenizer.decode(tail_ids, skip_special_tokens=True)


def extract_last_options_from_parser_llm_output(text: str) -> str:
    """
    Find the last occurrence of a multiple-choice answer (A–J) in the text. (from LLM output)
    Returns only the letter if found, otherwise returns an empty string.
    """
    matches = re.findall(r'Answer:\s*([A-J])\b', text)
    return matches[-1] if matches else ""




def build_parser_prompt(question: str, reasoning_trace: str) -> str:
    return rf"""You are an answer extractor.

Inputs:
- Question: {question}
- Reasoning trace: {reasoning_trace}

Task:
1) Read the Question and determine the expected final answer type. 
- Possible types include: Numeric scalar, Comparison/Ordering among variables, Set/List, Interval/Inequality, Coordinate/Tuple, Algebraic expression, or Multiple-choice letter.
- Decide the most appropriate type for THIS Question.

2) Carefully scan the Reasoning trace and identify the final/conclusive answer consistent with the expected type.
- Prefer the final/most conclusive statement (e.g., “Therefore…”, “Thus…”, “Final answer…”, or the last decisive equation).
- If multiple candidates appear, choose the last one that is self-consistent.
- Ignore exploratory or contradicted intermediate guesses.

3) Output EXACTLY in the format: \boxed{{FINAL_ANSWER}}

Formatting rules:
- Put ONLY the final answer inside \boxed{{}} (no units, words, or explanations).
- Do not include any explanation or extra symbols outside \boxed{{}}.
- If no conclusive final answer is present in the trace, choose the last consistent candidate stated as final; if still impossible, output \boxed{{NO_ANSWER}}.

Output:
"""

def build_option_parser_prompt(options_block: str, reasoning_trace: str) -> str:
    return rf"""You are an answer extractor.

You will be provided with the following inputs:
- Multiple-choice options (corresponding to the Question)
- A reasoning trace that shows the step-by-step thought process

Task:
1) Carefully scan the Reasoning trace and identify the final multiple-choice option answer.
- Valid answers are only single capital letters from [A-J].
- If the final answer in the Reasoning trace is given as option text instead of a letter, use the provided multiple-choice options to map it to the corresponding letter from [A-J].
- Prefer the final/most conclusive statement (e.g., "Therefore...", "Thus...", "Final answer...", or the last decisive choice).
- If multiple candidates appear, choose the last consistent one.
- Ignore exploratory or contradicted intermediate guesses.

2) Output EXACTLY in the format:
Answer: X

Formatting rules:
- Replace X with the chosen letter from [A-J].
- Do not include any explanation, units, or extra text.

Now, the inputs are given below.

Inputs:
- Multiple-choice options (corresponding to the Question): {options_block}
- Reasoning trace: {reasoning_trace}

Output:
"""



def to_chat_format(prompt, tokenizer) -> str:

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )   
    return chat_text


def extract_answers_from_text_list(llm, tokenizer, question_list, text_list, last_n_tokens=512):
    """
    Pipeline:
    1) LLM-based extraction using a parser prompt
       built from (Question, last_n_tokens of text).
    2) Return a list aligned with inputs, each being either the found \boxed{...}
       or an empty string if neither direct extraction nor LLM extraction yields one.
    """
    assert len(question_list) == len(text_list), "question_list and text_list must have the same length."

    n = len(text_list)
    extracted = [""] * n

    indices_needing_llm = []
    for i, text in enumerate(text_list):
        indices_needing_llm.append(i)

    if not indices_needing_llm:
        return extracted

    # Step 2: build prompts ONLY for those that still need LLM extraction
    trimmed_text_list = []
    for i in indices_needing_llm:
        text = text_list[i]
        trimmed = take_last_n_tokens(text, tokenizer, last_n_tokens)
        trimmed_text_list.append(trimmed)

    chat_prompts = []
    for qi, trimmed_text in zip(indices_needing_llm, trimmed_text_list):
        q = question_list[qi]
        prompt = build_parser_prompt(q, trimmed_text)
        chat_prompts.append(to_chat_format(prompt, tokenizer))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        seed=32,
    )

    print(f"Extracting answers via LLM for {len(chat_prompts)} out of {n} samples...")
    # Step 3: LLM inference for the subset
    responses = llm.generate(chat_prompts, sampling_params=sampling)

    # Step 4: parse LLM outputs for \boxed{...}; keep empty string if still missing
    for idx, response in zip(indices_needing_llm, responses):
        text = response.outputs[0].text if response.outputs and response.outputs[0].text else ""
        # Prefer explicit \boxed in the LLM output if present
        boxed = extract_last_boxed(text)
        # If model did not include \boxed, keep empty string (strict formatting)
        extracted[idx] = boxed if boxed!= "" else ""

    return extracted


def extract_option_answers_from_text_list(llm, tokenizer, options_list, text_list, last_n_tokens=512):
    """
    Pipeline:
    1) LLM-based extraction using a parser prompt
       built from (Options, last_n_tokens of text).
    3) Return a list aligned with inputs, each being either the option [A-J]
       or an empty string if neither direct extraction nor LLM extraction yields one.
    """
    assert len(options_list) == len(text_list), "options_list and text_list must have the same length."

    n = len(text_list)
    extracted = [""] * n

    random.seed(32)
    indices_needing_llm = []
    for i, text in enumerate(text_list):
        indices_needing_llm.append(i)

    # Step 2: build prompts ONLY for those that still need LLM extraction
    trimmed_text_list = []
    for i in indices_needing_llm:
        text = text_list[i]
        trimmed = take_last_n_tokens(text, tokenizer, last_n_tokens)
        trimmed_text_list.append(trimmed)

    chat_prompts = []
    for oi, trimmed_text in zip(indices_needing_llm, trimmed_text_list):
        options_block_in_str_format = options_list[oi]
        prompt = build_option_parser_prompt(options_block_in_str_format, trimmed_text)
        chat_prompts.append(to_chat_format(prompt, tokenizer))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        seed=32,
    )

    print(f"Extracting answers via LLM for {len(chat_prompts)} out of {n} samples...")
    # Step 3: LLM inference for the subset
    responses = llm.generate(chat_prompts, sampling_params=sampling)

    # Step 4: parse LLM outputs for extracting options; keep empty string if still missing
    for idx, response in zip(indices_needing_llm, responses):
        text = response.outputs[0].text if response.outputs and response.outputs[0].text else ""
        # Prefer explicit option in the LLM output if present
        option = extract_last_options_from_parser_llm_output(text)
        # Randomly chosen character from A-J if model did not include option letter
        extracted[idx] = option if option != "" else random.choice(list("ABCDEFGHIJ"))
    return extracted