import fasttext
import re
import unicodedata
from collections import Counter

import re
import unicodedata
from typing import List

# -----------------------------
# 1) LaTeX / code masking (replace with spaces)
# -----------------------------
_LATEX_BLOCKS = [
    (r"\$\$(.*?)\$\$", re.DOTALL),            # $$ ... $$
    (r"\\\[(.*?)\\\]", re.DOTALL),            # \[ ... \]
    (r"\\begin\{(equation|align\*?|eqnarray\*?)\}(.*?)\\end\{\1\}", re.DOTALL),
]
_LATEX_INLINE = [
    (r"(?<!\$)\$(?!\$)(.*?)(?<!\\)\$", re.DOTALL),  # $ ... $
    (r"\\\((.*?)\\\)", re.DOTALL),                  # \( ... \)
]
_CODE_BLOCKS = [
    (r"```.*?```", re.DOTALL),  # fenced code block
    (r"`[^`]*`", 0),            # inline code
]

def _strip_formulas_and_code(text: str) -> str:
    """Mask LaTeX and code spans with spaces so punctuation inside them isn't treated as sentence boundaries."""
    s = text
    for pat, flags in _CODE_BLOCKS + _LATEX_BLOCKS + _LATEX_INLINE:
        s = re.sub(pat, " ", s, flags=flags)
    return s

# -----------------------------
# 2) Heuristic to detect formula-like sentences
# -----------------------------
MATH_CATEGORIES = {"Sm", "Sk"}               # Unicode categories for math/modifier symbols
OP_CHARS = set("+-*/=<>^_{}[]()|\\")         # Common math/operator characters

def looks_like_formula(sentence: str,
                       alpha_thresh: float = 0.35,
                       op_min: int = 2,
                       digit_ratio_min: float = 0.20) -> bool:
    """
    Return True if the sentence looks like a math formula.
    Signals: low alphabetic ratio + high operator/digit density or explicit math symbols.
    LaTeX/code were already stripped to spaces, so no mask token is used here.
    """
    if not sentence:
        return False

    n = len(sentence)
    alpha = sum(ch.isalpha() for ch in sentence)
    digits = sum(ch.isdigit() for ch in sentence)
    ops = sum(1 for ch in sentence if ch in OP_CHARS)
    math_sym = any(unicodedata.category(ch) in MATH_CATEGORIES for ch in sentence)

    alpha_ratio = (alpha / n) if n else 0.0
    digit_ratio = (digits / n) if n else 0.0

    # Strong math signal: explicit math symbols present
    if math_sym and (alpha_ratio < alpha_thresh or ops >= 1 or digit_ratio >= digit_ratio_min):
        return True

    # Operator/digit density signals
    if ops >= op_min:
        return True
    if ops >= 1 and digit_ratio >= digit_ratio_min and alpha_ratio < 0.5:
        return True

    # Low alphabetic ratio combined with some numeric/operator content
    if alpha_ratio < alpha_thresh and (digit_ratio >= digit_ratio_min or ops >= 1):
        return True

    return False

# -----------------------------
# 3) Sentence segmentation (punctuation-only) + drop formula-like sentences
# -----------------------------
_SENT_PUNCT = r"(?:\.\.\.|…|[.?!]|[。？！]|؟|।|॥)"
_ABBR_TAIL = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Sra|Srta|St|No|etc|e\.g|i\.e|z\.B|u\.a|u\.Ä|z\.T|p\.ej|approx|ca)\.\s*$",
    re.IGNORECASE,
)

def _sentence_segments_by_punct(text: str,
                                drop_formula_like: bool = True,
                                alpha_thresh: float = 0.35,
                                min_len: int = 3) -> List[str]:
    """
    Split 'text' into sentences using punctuation only (no newline boundaries).
    Steps:
      1) Strip LaTeX/code spans (replaced with spaces) so their inner punctuation won't split sentences.
      2) Split by sentence punctuation (ellipsis/fullwidth included) while guarding decimals and abbreviations.
      3) Optionally drop sentences that look like formulas.

    Args:
        drop_formula_like: if True, remove sentences that look like formulas.
        alpha_thresh: threshold for alphabetic ratio used in looks_like_formula().
        min_len: if > 0, drop very short sentences after trimming. By default, we use min_len=3

    Returns:
        A list of natural-language sentences.
    """
    masked = _strip_formulas_and_code(text)
    # Normalize whitespace; do NOT treat newlines as boundaries
    t = re.sub(r"\s+", " ", masked).strip()
    if not t:
        return []

    # 1) First pass: keep delimiters to reconstruct sentences
    parts = re.split(f"({_SENT_PUNCT})", t)
    chunks = []
    buf = ""
    for i, p in enumerate(parts):
        if i % 2 == 0:  # content
            buf += p
        else:           # punctuation
            prev = buf.rstrip()
            # Decimal guard: don't split on '.' when it ends with \d.\d
            if p == "." and re.search(r"\d\.\d$", prev):
                buf += p
            else:
                seg = (prev + p).strip()
                if seg:
                    chunks.append(seg)
                buf = ""
    tail = buf.strip()
    if tail:
        chunks.append(tail)

    # 2) Abbreviation merge: merge 'Dr.' + next chunk, etc.
    merged = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if _ABBR_TAIL.search(cur) and i + 1 < len(chunks):
            cur = (cur + " " + chunks[i + 1]).strip()
            i += 2
        else:
            i += 1
        merged.append(cur)

    # 3) Optional drops: formula-like and too-short sentences
    out = []
    for s in merged:
        if min_len and len(s.strip()) < min_len:
            continue
        if drop_formula_like and looks_like_formula(s, alpha_thresh=alpha_thresh):
            continue
        out.append(s)

    return out



def lid(text, fasttext_model): return fasttext_model.predict(text.replace("\n"," "), k=1)[0][0].replace("__label__","")

def get_lid_distribution(text, fasttext_model, min_len=3):
    # Remove LaTeX and formula-like content
    text_lines = _sentence_segments_by_punct(text, drop_formula_like=True, min_len=min_len)
    if text_lines == []:
        # Sometimes, the entire text is formula-like and gets removed
        return {}
    lang_ids = [lid(line, fasttext_model) for line in text_lines]
    # Count occurrences of each language ID
    lang_counts = {}
    for lang_id in lang_ids:
        if lang_id in lang_counts:
            lang_counts[lang_id] += 1
        else:
            lang_counts[lang_id] = 1
    return lang_counts

def compute_language_id_statistics(text, fasttext_model):
    text_lid_dist = get_lid_distribution(text, fasttext_model)

    if text_lid_dist == {}:
        return {} # If no valid text remains after filtering, return empty dict
    total_count = sum(text_lid_dist.values())
    if total_count == 0:
        return {lid: 0 for lid in text_lid_dist.keys()}
    
    lid_percentages = [(lid, (count / total_count) * 100) for lid, count in text_lid_dist.items()]
    # Sort percentages by language ID
    lid_percentages.sort(key=lambda x: x[1], reverse=True)  # Sort by percentage

    return lid_percentages
    



def compute_language_id_statistics_self_english_and_other(
    res, 
    lang_id, 
    fasttext_model, 
    sort_desc=True,
    round_digits=5
):
    """
    Return per-language percentage breakdowns for:
      - total_percentages
      - pred_percentages
      - reasoning_percentages

    Behavior:
      - If the underlying LID distribution is empty ({}), the corresponding output dict is {}.
      - Otherwise, include "self" (mapped from lang_id) and "en" keys (0.0 if absent), plus all other detected lids.
    """

    prediction = res.get("prediction", "")
    reasoning_trace = res.get("reasoning_trace", "")

    # Normalize zh-cn -> zh for "self" mapping
    if lang_id == "zh-cn":
        lang_id = "zh"

    # These should return dict-like objects; may be {} when no valid segments are detected
    pred_lid_dist = get_lid_distribution(prediction, fasttext_model)
    reasoning_lid_dist = get_lid_distribution(reasoning_trace, fasttext_model)
    # Accumulate counts for "total" = prediction + reasoning
    total_counts = Counter()
    if pred_lid_dist:
        total_counts.update(pred_lid_dist)
    if reasoning_lid_dist:
        total_counts.update(reasoning_lid_dist)

    def to_percentages(counts):
        """Convert counts to percentage dict; {} -> {}."""
        if not counts:
            return {}
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: (v / total) * 100.0 for k, v in counts.items()}

    total_pct = to_percentages(total_counts)
    pred_pct = to_percentages(pred_lid_dist)
    reasoning_pct = to_percentages(reasoning_lid_dist)


    def build_selected(pct_dict):
        """
        When pct_dict == {}, return {}.
        Otherwise:
          - include "self" and "en" (0.0 if missing),
          - add all other lids as-is,
          - optionally sort others by desc value while keeping self/en at the top.
        """
        if not pct_dict:
            return {}

        out = {}
        # Always expose "self" and "en" when there is any signal
        out["self"] = round(pct_dict.get(lang_id, 0.0), round_digits)
        out["en"] = round(pct_dict.get("en", 0.0), round_digits)
        # Add every other detected language individually
        for lid, val in pct_dict.items():
            if lid not in (lang_id, "en"):
                out[lid] = round(val, round_digits)

        if sort_desc:
            fixed = {"self": out["self"], "en": out["en"]}
            others = {k: v for k, v in out.items() if k not in ("self", "en")}
            others_sorted = dict(sorted(others.items(), key=lambda x: x[1], reverse=True))
            return {**fixed, **others_sorted}
        return out

    return {
        "total_percentages": build_selected(total_pct),
        "pred_percentages": build_selected(pred_pct),
        "reasoning_percentages": build_selected(reasoning_pct),
    }



def compute_language_id_statistics_per_language(res, lang_id, chunk_size=20):
    """
    Compute per-language percentage breakdowns for:
      - total_percentages  (prediction + reasoning)
      - pred_percentages   (prediction only)
      - reasoning_percentages (reasoning only)

    Behavior on empties:
      - If get_lid_distribution(...) returns {}, the corresponding percentage dict is {}.
      - If both prediction and reasoning distributions are {}, all three outputs are {}.
    """
    # Safe text extraction
    prediction = res.get("prediction") or ""
    reasoning_trace = res.get("reasoning_trace") or ""

    # Normalize zh-cn -> zh for "self" mapping
    if lang_id == "zh-cn":
        lang_id = "zh"

    # May return {}; guard with "or {}"
    pred_lid_dist = get_lid_distribution(prediction, chunk_size) 
    reasoning_lid_dist = get_lid_distribution(reasoning_trace, chunk_size) 

    # Early exit: nothing detected at all
    if pred_lid_dist == {} and reasoning_lid_dist == {}:
        return {
            "total_percentages": {},
            "pred_percentages": {},
            "reasoning_percentages": {},
        }

    # Aggregate counts for total = prediction + reasoning
    total_counts = Counter()
    if pred_lid_dist:
        total_counts.update(pred_lid_dist)
    if reasoning_lid_dist:
        total_counts.update(reasoning_lid_dist)

    def to_percentages(counts: dict) -> dict:
        """Convert counts -> percentages in [0,100); {} or zero-sum -> {}."""
        if not counts:
            return {}
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {lid: (cnt / total) * 100.0 for lid, cnt in counts.items()}

    total_percentages    = to_percentages(total_counts)
    pred_percentages     = to_percentages(pred_lid_dist)
    reasoning_percentages= to_percentages(reasoning_lid_dist)

    def replace_self(mapping: dict) -> dict:
        """
        Replace the key equal to lang_id with 'self'.
        If mapping == {}, return {} (do not force 'self' to appear).
        """
        if not mapping:
            return {}
        return {("self" if lid == lang_id else lid): perc
                for lid, perc in mapping.items()}

    return {
        "total_percentages": replace_self(total_percentages),
        "pred_percentages": replace_self(pred_percentages),
        "reasoning_percentages": replace_self(reasoning_percentages),
    }
