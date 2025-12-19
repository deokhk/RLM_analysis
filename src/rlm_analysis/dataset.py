# dataset.py
import json
import random 
import logging 
import copy
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from datasets import load_dataset
from sympy import Not 
from rlm_analysis.lang_libs import LANG_LIBS, LANG_SUBJECTS


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# We use it for MGSM as well. (In order to make the final answer boxed). Taken from Polymath
MATH_LANG_TO_FORMATTING_INSTRUCTION = {
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "zh": "注意：请将最终答案放在 $\\boxed{}$ 中。",
    "ar": "ملاحظة: يُرجى وضع الإجابة النهائية في $\\boxed{}$.",
    "bn": "বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{}$ এর মধ্যে রাখুন।",
    "de": "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$.",
    "es": "Nota: Por favor, coloque la respuesta final en el $\\boxed{}$.",
    "fr": "Remarque : Veuillez mettre la réponse finale dans le $\\boxed{}$.",
    "id": "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{}$.",
    "it": "Nota: Per favore, metti la risposta finale nel $\\boxed{}$.",
    "ja": "注意：最終的な答えを $\\boxed{}$ に入れてください。",
    "ko": "참고: 최종 답안을 $\\boxed{}$ 안에 넣어 주세요.",
    "ms": "Nota: Sila letakkan jawapan akhir dalam $\\boxed{}$.",
    "pt": "Nota: Por favor, coloque a resposta final no $\\boxed{}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$.",
    "sw": "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{}$.",
    "te": "గమనిక: దయచేసి తుది జవాబును $\\boxed{}$ లో ఉంచండి.",
    "th": "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed{}$.",
    "vi": "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{}$.",
    "mr": "टीप: कृपया अंतिम उत्तर $\boxed{}$ मध्ये लिहा.",
    "wo": "Note: Jëfandikoo $\boxed{}$ ngir bind tontu bi mujj."
}


MATH_LANG_TO_FORMATTING_INST_ANSWER_ONLY = {
    "en": "Important: Output ONLY the final answer, and put it inside $\\boxed{}$. Do not include any explanation or text outside the box.",
    "zh": "重要：只输出最终答案，并将其放在 $\\boxed{}$ 中。不要输出任何解释或额外文字。",
    "ar": "مهم: أخرج الإجابة النهائية فقط، وضعها داخل $\\boxed{}$. لا تكتب أي تفسير أو نص خارج الصندوق.",
    "bn": "গুরুত্বপূর্ণ: শুধুমাত্র চূড়ান্ত উত্তরটি $\\boxed{}$ এর মধ্যে লিখুন। এর বাইরে কোনো ব্যাখ্যা বা লেখা দেবেন না।",
    "de": "Wichtig: Gib NUR die endgültige Antwort aus und setze sie in $\\boxed{}$. Keine Erklärungen oder zusätzlichen Texte außerhalb der Box.",
    "es": "Importante: Solo escribe la respuesta final y colócala dentro de $\\boxed{}$. No incluyas ninguna explicación ni texto adicional.",
    "fr": "Important : Produisez UNIQUEMENT la réponse finale, dans $\\boxed{}$. N’ajoutez aucune explication ni texte en dehors de la boîte.",
    "id": "Penting: Hanya keluarkan jawaban akhir, dan letakkan di dalam $\\boxed{}$. Jangan sertakan penjelasan atau teks lain.",
    "it": "Importante: Fornisci SOLO la risposta finale e mettila in $\\boxed{}$. Non includere spiegazioni o testo aggiuntivo.",
    "ja": "重要：最終的な答えのみを $\\boxed{}$ に入れて出力してください。説明や余分なテキストは出力しないでください。",
    "ko": "중요: 오직 최종 답안만 출력하고 반드시 $\\boxed{}$ 안에 넣어 주세요. 그 외의 설명이나 텍스트는 출력하지 마세요.",
    "ms": "Penting: Hanya keluarkan jawapan akhir dan letakkan dalam $\\boxed{}$. Jangan sertakan sebarang penjelasan atau teks lain.",
    "pt": "Importante: Produza APENAS a resposta final, dentro de $\\boxed{}$. Não inclua explicações ou texto adicional.",
    "ru": "Важно: Выводите ТОЛЬКО окончательный ответ и помещайте его в $\\boxed{}$. Не добавляйте объяснений или лишнего текста.",
    "sw": "Muhimu: Toa jibu la mwisho PEKEE, na uliweke ndani ya $\\boxed{}$. Usiongeze maelezo au maandishi mengine.",
    "te": "ముఖ్యం: కేవలం తుది జవాబును మాత్రమే $\\boxed{}$ లో ఉంచండి. అదనపు వివరణ లేదా పాఠ్యం ఇవ్వవద్దు.",
    "th": "สำคัญ: แสดงเฉพาะคำตอบสุดท้าย และใส่ไว้ใน $\\boxed{}$ เท่านั้น ห้ามมีคำอธิบายหรือข้อความอื่นใด",
    "vi": "Quan trọng: Chỉ xuất ra đáp án cuối cùng và đặt trong $\\boxed{}$. Không thêm giải thích hay văn bản nào khác.",
}



LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "zh": "Chinese",
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "vi": "Vietnamese"
}



LANGUAGE_NAME_TO_CODE = {v: k for k, v in LANGUAGE_CODE_TO_NAME.items()}

class BaseDataset(ABC):
    """
    Base class for all dataset classes.

    In concrete implementations, the dataset can be loaded by:
    - Passing a file path during initialization, or
    - Implementing the load_data() method to read data from files.
    """

    def __init__(self):
        # Stores the raw loaded data
        self.data = []

    @abstractmethod
    def load_data(self):
        """
        Load data from files and store it in self.data.

        This method should handle all file I/O logic.
        The loaded data will later be processed and formatted
        by methods such as get_test_data().
        """
        pass

    @abstractmethod
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Return the test dataset grouped by language.

        Example:
            {
                "en": [...],
                "ko": [...],
                ...
            }
        """
        pass


class PolyMathDataset(BaseDataset):
    """
    PolyMath dataset:
    Loads the test set for each language specified by a list of language codes.
    The get_test_data() method returns the data as lists grouped by language code.
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {} 
        self.args = args
        self.split = args.polymath_split 
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()] 
        self.do_thinking_intervention = self.args.do_thinking_intervention

        self.load_data()

    def load_data(self):
        for lang in self.eval_langs:
            test_dataset = load_dataset("Qwen/PolyMath", lang, split=self.split)

            self.datasets[lang] = test_dataset


    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                id_idx = int(id.split("-")[-1])
                corresponding_eng_datapoint = self.datasets["en"][id_idx]
                # Sanity check
                assert id_idx == int(corresponding_eng_datapoint["id"].split("-")[-1]), f"ID mismatch: {id} vs {corresponding_eng_datapoint['id']}"
                english_question = corresponding_eng_datapoint["question"]

                formatted_question = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                if "Qwen3" in self.args.model_name or "gpt-oss" in self.args.model_name:
                    # By default, Qwen tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": formatted_question
                        }
                    ]
                    if self.args.thinking_intervention_lang == "en":
                        thinking_intervention_quote = f"Okay, let's see. I understand the question as: '{english_question}'. Let's solve the problem based on this understanding."
                    else:
                        raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                    if "Qwen3" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                else:
                    raise ValueError(f"Unsupported model name: {self.args.model_name}")
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "split": self.split,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data


# MMLU-ProX lite dataset 

class MMLUProXLiteDataset(BaseDataset):
    """
    MMLU-proX lite dataset:
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()]
        self.reasoning_categories = ["math", "physics", "chemistry", "computer science", "engineering"] 
        self.do_thinking_intervention = self.args.do_thinking_intervention

        # Parallel English index: question_id -> {"question": str, "options_string": str}
        self.en_by_id: Dict[str, Dict[str, str]] = {}

        self.load_data()

    # -------------------- Utilities --------------------
    @staticmethod
    def _collect_options_from_datapoint(datapoint: Dict[str, Any]) -> List[str]:
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

    @staticmethod
    def _format_options_string(options: List[str]) -> str:
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

    def load_data(self):
        # Load test set with reasoning categories only

        # 1) Build English index for parallel reference
        en_test = load_dataset("li-lab/MMLU-ProX-Lite", "en", split="test")
        en_reason = en_test.filter(lambda x: x["category"] in self.reasoning_categories)

        for dp in en_reason:
            qid = dp["question_id"]
            en_question = dp["question"]
            en_options = self._collect_options_from_datapoint(dp)
            en_options_string = self._format_options_string(en_options)
            self.en_by_id[qid] = {
                "question": en_question,
                "options_string": en_options_string
            }

        for lang in self.eval_langs:
            test_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="test")
            test_reasoning_dataset = test_dataset.filter(lambda x : x['category'] in self.reasoning_categories)

            self.datasets[lang] = test_reasoning_dataset
            print(f"Loaded {len(test_reasoning_dataset)} samples for language: {lang}")

    def get_prompt(self, question, category, options, lang):
        lang_lib_template = LANG_LIBS.get(lang, "")
        question_formatter = lang_lib_template[0]
        option_formatter = lang_lib_template[1]

        ans_suffix = lang_lib_template[5].format("X")
        subject_in_lang = ""
        if category == "computer science":
            # Use "computer_science" as the key if "computer science" is not found
            try:
                subject_in_lang = LANG_SUBJECTS[lang][category]
            except KeyError:
                subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
        else:
            subject_in_lang = LANG_SUBJECTS[lang][category]

        assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
        options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        options_string = ""
        for idx, option in enumerate(options):
            options_string += f"({options_letters[idx]}) {option}\n"
        options_string = options_string.rstrip("\n")
        prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
        return prompt 

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["question_id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                options_dict = {}
                options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                options = [datapoint[field] for field in option_fields if field in datapoint and (datapoint[field] != None and datapoint[field] != "N/A")]

                for idx, option in enumerate(options):
                    options_dict[options_letters[idx]] = option
                prompt = self.get_prompt(question, datapoint["category"], options, lang)

                if self.args.thinking_intervention_lang == "en":
                    en_question = self.en_by_id[id]["question"]
                    en_options_string = self.en_by_id[id]["options_string"]
                    thinking_intervention_quote = f"Okay, let's see. I understand the question as: {en_question}', and options as: {en_options_string}. Let's solve the problem based on this understanding."
                else:
                    raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                if "gpt-oss" in self.args.model_name or "Qwen3" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                else:
                    raise ValueError(f"Unsupported model for MMLU-ProX lite: {self.args.model_name}")
                
                if self.do_thinking_intervention:
                    if "Qwen3" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict,
                    "options_dict": options_dict
                })
            all_test_data[lang] = dataset_list

        return all_test_data


# =========================================================================
# For other experiments..
#                       /^--^\     /^--^\     /^--^\
#                       \____/     \____/     \____/
#                      /      \   /      \   /      \
#                     |        | |        | |        |
#                      \__  __/   \__  __/   \__  __/
# |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
# | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
# | | | | | | | | | | | | / / | | |\ \| | | | | |/ /| | | | | | | | | | | |
# | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
# #########################################################################
# | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
# =========================================================================


class LowResourcePolyMathDataset(BaseDataset):
    """
    LowResource PolyMath dataset:
    Used for low-resource languages where translated questions are provided in a JSON file.
    (not present in the original PolyMath dataset)
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.split = args.polymath_split 
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()] 
        self.do_thinking_intervention = self.args.do_thinking_intervention
        self.translated_dataset_json_path = args.translated_dataset_json_path
        self.load_data()

    def load_data(self):
        # Load the translated dataset from the provided JSON file
        with open(self.translated_dataset_json_path, "r", encoding="utf-8") as f:
            translated_dataset = json.load(f)

        # Iterate over the dataset, and remap the id 
        dataset_languages = []
        for lang, language_dataset in translated_dataset.items():
            for idx, datapoint in enumerate(language_dataset):
                datapoint_original_id = datapoint["id"]
                datapoint["id"] = datapoint_original_id.replace("en", lang)
            self.datasets[lang] = translated_dataset[lang]
            dataset_languages.append(lang)
        # Sanity check 
        eval_langs_given = self.args.eval_langs.split(",")
        assert set(eval_langs_given) == set(dataset_languages), "Some eval_langs are not found in the translated dataset!"

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["id"]
                original_english_question = datapoint["question"]
                translated_question = datapoint["translated_question"]
                answer = datapoint["answer"]

                formatted_question = translated_question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                if "Qwen3" in self.args.model_name or "gpt-oss" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": formatted_question
                        }
                    ]
                    if self.args.thinking_intervention_lang == "en":
                        thinking_intervention_quote = f"Okay, let's see. I understand the question as: '{original_english_question}'. Let's solve the problem based on this understanding."
                    else:
                        raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                    if "Qwen3" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                else:
                    raise ValueError(f"Unsupported model name: {self.args.model_name}")
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "split": self.split,
                    "question": translated_question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data


class TranslatedPolyMathDataset(BaseDataset):
    """
    TranslatedPolyMath dataset:
    For each specified language code, the corresponding test set is loaded,  
    and when get_test_data() is called, it returns a list of data items per language code.  

    (Since all languages solve the same set of problems, a pre-translated JSON file with English questions is used.)
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.split = args.polymath_split 
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()] 
        self.translated_dataset_json_path = args.translated_dataset_json_path

        assert args.polymath_split in args.translated_dataset_json_path, "Split mismatch between args and translated dataset json path!"
        self.load_data()

    def load_data(self):
        # 각 언어별로 test set만 저장합니다.
        for lang in self.eval_langs:
            test_dataset = load_dataset("Qwen/PolyMath", lang, split=self.split)
            self.datasets[lang] = test_dataset
        
        print(f"Loading translated PolyMath dataset from {self.translated_dataset_json_path}")
        with open(self.translated_dataset_json_path, "r", encoding="utf-8") as f:
            translated_dataset = json.load(f)  
        
        self.translated_dataset = translated_dataset


    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                id_idx = int(id.split("-")[-1])
                if lang == "en":
                    # In case of English, use the original question
                    corresponding_translated_datapoint = {
                        "id": id, 
                        "translated_question": question
                    }
                else:
                    corresponding_translated_datapoint = self.translated_dataset[lang][id_idx]
                # Sanity check
                assert id_idx == int(corresponding_translated_datapoint["id"].split("-")[-1]), f"ID mismatch: {id} vs {corresponding_translated_datapoint['id']}"
                english_question = corresponding_translated_datapoint["translated_question"]

                english_formatted_question = english_question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION["en"] + f"\nPlease respond in {LANGUAGE_CODE_TO_NAME[lang]}."
                prompt_dict = [
                    {
                        "role": "user",
                        "content": english_formatted_question
                    }
                ]
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "split": self.split,
                    "question": english_question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data

class TranslatedThinkIntvPolyMathDataset(BaseDataset):
    """
    TranslatedThinkIntvPolyMath dataset:
    For each specified language code, the corresponding test set is loaded,  
    and when get_test_data() is called, it returns a list of data items per language code.  

    (Since all languages solve the same set of problems, a pre-translated JSON file with English questions is used.)
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.split = args.polymath_split 
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()] 
        self.translated_dataset_json_path = args.translated_dataset_json_path

        assert args.polymath_split in args.translated_dataset_json_path, "Split mismatch between args and translated dataset json path!"
        self.load_data()

    def load_data(self):
        # 각 언어별로 test set만 저장합니다.
        for lang in self.eval_langs:
            test_dataset = load_dataset("Qwen/PolyMath", lang, split=self.split)
            self.datasets[lang] = test_dataset
        
        print(f"Loading translated PolyMath dataset from {self.translated_dataset_json_path}")
        with open(self.translated_dataset_json_path, "r", encoding="utf-8") as f:
            translated_dataset = json.load(f)  
        
        self.translated_dataset = translated_dataset


    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                id_idx = int(id.split("-")[-1])
                if lang == "en":
                    # In case of English, use the original question
                    corresponding_translated_datapoint = {
                        "id": id, 
                        "translated_question": question
                    }
                else:
                    corresponding_translated_datapoint = self.translated_dataset[lang][id_idx]
                # Sanity check
                assert id_idx == int(corresponding_translated_datapoint["id"].split("-")[-1]), f"ID mismatch: {id} vs {corresponding_translated_datapoint['id']}"
                english_question = corresponding_translated_datapoint["translated_question"]

                formatted_question = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                if "Qwen3" in self.args.model_name or "gpt-oss" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": formatted_question
                        }
                    ]
                    if self.args.thinking_intervention_lang == "en":
                        thinking_intervention_quote = f"Okay, let's see. I understand the question as: '{english_question}'. Let's solve the problem based on this understanding."
                    else:
                        raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                    if "Qwen3" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                else:
                    raise ValueError(f"Unsupported model name: {self.args.model_name}")
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "split": self.split,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data


class DifferentIntvPolyMathDataset(PolyMathDataset):
    """
    DifferentIntvPolyMath dataset:
    Loads the test set for each language specified by a list of language codes.
    The get_test_data() method returns the data as lists grouped by language code.
    - The thinking intervention quote is varied based on the specified type.
    """
    def __init__(self, args):
        super().__init__(args)
        self.thinking_intv_quote_type = args.thinking_intervention_quote_type


    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                id_idx = int(id.split("-")[-1])
                corresponding_eng_datapoint = self.datasets["en"][id_idx]
                # Sanity check
                assert id_idx == int(corresponding_eng_datapoint["id"].split("-")[-1]), f"ID mismatch: {id} vs {corresponding_eng_datapoint['id']}"
                english_question = corresponding_eng_datapoint["question"]

                formatted_question = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                if "Qwen3" in self.args.model_name or "gpt-oss" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": formatted_question
                        }
                    ]
                    if self.args.thinking_intervention_lang == "en":
                        if self.args.thinking_intervention_quote_type == "default":
                            thinking_intervention_quote = f"Okay, let's see. I understand the question as: {english_question}'. Let's solve the problem based on this understanding."
                        elif self.args.thinking_intervention_quote_type == "simple":
                            thinking_intervention_quote = f"English meaning of the question: '{english_question}'. I'll solve the problem based on this understanding."
                        elif self.args.thinking_intervention_quote_type == "paraphrase_a":
                            thinking_intervention_quote = f"Okay, I understand the question as: '{english_question}'. I will solve the problem based on this understanding."
                        elif self.args.thinking_intervention_quote_type == "paraphrase_b":
                            thinking_intervention_quote = f"Okay, my understanding of the question in English is: '{english_question}'. I will proceed using this interpretation."

                    else:
                        raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                    if "Qwen3" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name and self.do_thinking_intervention:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                else:
                    raise ValueError(f"Unsupported model name: {self.args.model_name}")
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "split": self.split,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data




class DifferentIntvMMLUProXLiteDataset(MMLUProXLiteDataset):
    def __init__(self, args):
        super().__init__(args)
        self.thinking_intv_quote_type = args.thinking_intervention_quote_type

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["question_id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                options_dict = {}
                options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                options = [datapoint[field] for field in option_fields if field in datapoint and (datapoint[field] != None and datapoint[field] != "N/A")]

                for idx, option in enumerate(options):
                    options_dict[options_letters[idx]] = option
                prompt = self.get_prompt(question, datapoint["category"], options, lang)

                if self.args.thinking_intervention_lang == "en":
                    en_question = self.en_by_id[id]["question"]
                    en_options_string = self.en_by_id[id]["options_string"]
                    if self.args.thinking_intervention_quote_type == "default":
                        thinking_intervention_quote = f"Okay, let's see. I understand the question as: {en_question}', and options as: {en_options_string}. Let's solve the problem based on this understanding."
                    elif self.args.thinking_intervention_quote_type == "simple":
                        thinking_intervention_quote = f"English meaning of the question: '{en_question}'. English meaning of options: {en_options_string}. I'll solve the problem based on this understanding."
                    elif self.args.thinking_intervention_quote_type == "paraphrase_a":
                        thinking_intervention_quote = f"Okay, I understand the question as: '{en_question}', and options as: {en_options_string}. I will solve the problem based on this understanding."
                    elif self.args.thinking_intervention_quote_type == "paraphrase_b":
                        thinking_intervention_quote = f"My understanding of the question in English is: '{en_question}', and of the options is: {en_options_string}. I will proceed using this interpretation."
                else:
                    raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")

                if "gpt-oss" in self.args.model_name or "Qwen3" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                else:
                    raise ValueError(f"Unsupported model for MMLU-ProX lite: {self.args.model_name}")
                
                if self.do_thinking_intervention:
                    if "Qwen3" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict,
                    "options_dict": options_dict
                })
            all_test_data[lang] = dataset_list

        return all_test_data



class TranslatedMMLUProXLiteDataset(BaseDataset):
    """
    MMLU-proX lite dataset:
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()]
        self.reasoning_categories = ["math", "physics", "chemistry", "computer science", "engineering"] 
        self.translated_dataset_json_path = args.translated_dataset_json_path
        self.translated_data_lang_by_id = {}
        self.load_data()

    # -------------------- Utilities --------------------
    @staticmethod
    def _collect_options_from_datapoint(datapoint: Dict[str, Any]) -> List[str]:
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

    @staticmethod
    def _format_options_string(options: List[str]) -> str:
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

    def load_data(self):
        # Load test set with reasoning categories only

        for lang in self.eval_langs:
            test_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="test")
            test_reasoning_dataset = test_dataset.filter(lambda x : x['category'] in self.reasoning_categories)

            self.datasets[lang] = test_reasoning_dataset
            print(f"Loaded {len(test_reasoning_dataset)} samples for language: {lang}")

        print(f"Loading translated mmlu-prox-lite dataset from {self.translated_dataset_json_path}")
        with open(self.translated_dataset_json_path, "r", encoding="utf-8") as f:
            translated_dataset = json.load(f)  
        
        for lang, lang_dataset in translated_dataset.items():
            for dp in lang_dataset:
                qid = dp["question_id"]
                if lang not in self.translated_data_lang_by_id:
                    self.translated_data_lang_by_id[lang] = {}
                self.translated_data_lang_by_id[lang][qid] = {
                    "translated_question": dp["translated_question"],
                    "translated_options_string": dp["translated_options_string"]
                }


    def get_prompt(self, question, category, options, lang, options_string=None):
        lang_lib_template = LANG_LIBS.get(lang, "")
        question_formatter = lang_lib_template[0]
        option_formatter = lang_lib_template[1]

        ans_suffix = lang_lib_template[5].format("X")
        subject_in_lang = ""
        if category == "computer science":
            # Use "computer_science" as the key if "computer science" is not found
            try:
                subject_in_lang = LANG_SUBJECTS[lang][category]
            except KeyError:
                subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
        else:
            subject_in_lang = LANG_SUBJECTS[lang][category]

        assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
        if options_string is None:
            options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            options_string = ""
            for idx, option in enumerate(options):
                options_string += f"({options_letters[idx]}) {option}\n"
            options_string = options_string.rstrip("\n")
        prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
        return prompt 

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["question_id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                options_dict = {}
                options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                options = [datapoint[field] for field in option_fields if field in datapoint and (datapoint[field] != None and datapoint[field] != "N/A")]

                for idx, option in enumerate(options):
                    options_dict[options_letters[idx]] = option
                if lang == "en":
                    # In case of English, use the original question and options
                    en_question = question
                    en_options_string = self._format_options_string(options)
                else:
                    en_question = self.translated_data_lang_by_id[lang][id]["translated_question"]
                    en_options_string = self.translated_data_lang_by_id[lang][id]["translated_options_string"]
                prompt = self.get_prompt(en_question, datapoint["category"], None, "en", options_string=en_options_string)
                prompt_dict = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "question": en_question,
                    "answer": answer,
                    "prompt_dict": prompt_dict,
                    "options_dict": options_dict
                })
            all_test_data[lang] = dataset_list

        return all_test_data


class TranslatedThinkIntvMMLUProXLiteDataset(BaseDataset):
    """
    MMLU-proX lite dataset:
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()]
        self.reasoning_categories = ["math", "physics", "chemistry", "computer science", "engineering"] 
        self.translated_dataset_json_path = args.translated_dataset_json_path
        self.translated_data_lang_by_id = {}
        self.load_data()

    # -------------------- Utilities --------------------
    @staticmethod
    def _collect_options_from_datapoint(datapoint: Dict[str, Any]) -> List[str]:
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

    @staticmethod
    def _format_options_string(options: List[str]) -> str:
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

    def load_data(self):
        # Load test set with reasoning categories only

        for lang in self.eval_langs:
            test_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="test")
            test_reasoning_dataset = test_dataset.filter(lambda x : x['category'] in self.reasoning_categories)

            self.datasets[lang] = test_reasoning_dataset
            print(f"Loaded {len(test_reasoning_dataset)} samples for language: {lang}")

        print(f"Loading translated mmlu-prox-lite dataset from {self.translated_dataset_json_path}")
        with open(self.translated_dataset_json_path, "r", encoding="utf-8") as f:
            translated_dataset = json.load(f)  
        
        for lang, lang_dataset in translated_dataset.items():
            for dp in lang_dataset:
                qid = dp["question_id"]
                if lang not in self.translated_data_lang_by_id:
                    self.translated_data_lang_by_id[lang] = {}
                self.translated_data_lang_by_id[lang][qid] = {
                    "translated_question": dp["translated_question"],
                    "translated_options_string": dp["translated_options_string"]
                }


    def get_prompt(self, question, category, options, lang):
        lang_lib_template = LANG_LIBS.get(lang, "")
        question_formatter = lang_lib_template[0]
        option_formatter = lang_lib_template[1]

        ans_suffix = lang_lib_template[5].format("X")
        subject_in_lang = ""
        if category == "computer science":
            # Use "computer_science" as the key if "computer science" is not found
            try:
                subject_in_lang = LANG_SUBJECTS[lang][category]
            except KeyError:
                subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
        else:
            subject_in_lang = LANG_SUBJECTS[lang][category]

        assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
        options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        options_string = ""
        for idx, option in enumerate(options):
            options_string += f"({options_letters[idx]}) {option}\n"
        options_string = options_string.rstrip("\n")
        prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
        return prompt 

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["question_id"]
                question = datapoint["question"]
                answer = datapoint["answer"]
                options_dict = {}
                options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                options = [datapoint[field] for field in option_fields if field in datapoint and (datapoint[field] != None and datapoint[field] != "N/A")]

                for idx, option in enumerate(options):
                    options_dict[options_letters[idx]] = option
                prompt = self.get_prompt(question, datapoint["category"], options, lang)
                if lang == "en":
                    # In case of English, use the original question and options
                    en_question = question
                    en_options_string = self._format_options_string(options)
                else:
                    en_question = self.translated_data_lang_by_id[lang][id]["translated_question"]
                    en_options_string = self.translated_data_lang_by_id[lang][id]["translated_options_string"]
                thinking_intervention_quote = f"Okay, let's see. I understand the question as: {en_question}', and options as: {en_options_string}. Let's solve the problem based on this understanding."

                if "gpt-oss" in self.args.model_name or "Qwen3" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    if "Qwen3" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": "<think>\n" + thinking_intervention_quote
                        })
                    elif "gpt-oss" in self.args.model_name:
                        prompt_dict.append({
                            "role": "assistant",
                            "content": thinking_intervention_quote
                        })

                else:
                    raise ValueError(f"Unsupported model for MMLU-ProX lite: {self.args.model_name}")
                

                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict,
                    "options_dict": options_dict
                })
            all_test_data[lang] = dataset_list

        return all_test_data



# =========================================================================
# For "understandability" analysis
#                       /^--^\     /^--^\     /^--^\
#                       \____/     \____/     \____/
#                      /      \   /      \   /      \
#                     |        | |        | |        |
#                      \__  __/   \__  __/   \__  __/
# |^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
# | | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
# | | | | | | | | | | | | / / | | |\ \| | | | | |/ /| | | | | | | | | | | |
# | | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
# #########################################################################
# | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
# =========================================================================


class FilteredMGSMDatasetForCalibration(BaseDataset):
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.do_thinking_intervention = self.args.do_thinking_intervention
        self.load_data()

    def load_data(self):
        language_codes = ["en", "bn", "de", "es", "ar", "ja", "ko", "sw", "te", "th"]
        for lang in language_codes:
            print(f"Loading MGSM dataset for language: {lang}")
            ds = load_dataset("deokhk/filtered_mgsm_with_ids", split=lang)
            # 각 언어별로 test set만 저장합니다.
            self.datasets[lang] = ds

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        for lang, test_dataset in self.datasets.items():
            dataset_list = []
            for idx, sample in enumerate(test_dataset):
                question = sample["question"]
                answer = sample["answer_number"]
                # 각 언어에 맞는 prompt를 사용합니다.
                prompt = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                if "gpt-oss" in self.args.model_name or "Qwen3" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                    if self.do_thinking_intervention:
                        if self.args.thinking_intervention_lang == "en":
                            en_question = sample["question_en"]
                            thinking_intervention_quote = f"Okay, let's see. I understand the question as: '{en_question}'. Let's solve the problem based on this understanding."
                        else:
                            raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")
                        if "gpt-oss" in self.args.model_name:
                            # Here, gpt-oss use "harmony" format which does not use "<think>"
                            prompt_dict.append({
                                "role": "assistant",
                                "content": thinking_intervention_quote
                            })
                        else:
                            prompt_dict.append({
                                "role": "assistant",
                                "content": "<think>\n" + thinking_intervention_quote
                            })
                else:
                    raise ValueError(f"Unsupported model for Filtered MGSM: {self.args.model_name}")

                dataset_list.append({
                    "id": idx,
                    "language_code": lang,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict
                })
            all_test_data[lang] = dataset_list
        return all_test_data

class MMLUProXLiteDatasetForCalibration(BaseDataset):
    """
    MMLU-proX lite dataset for Calibration (which will be later used for understandability analysis):
    We use "dev" split here to get the model calibration results.
    This object support thinking intervention.
    """
    def __init__(self, args):
        super().__init__()
        self.datasets = {}  # 언어 코드별로 데이터를 저장할 dict
        self.args = args
        self.eval_langs = [lang.strip() for lang in list(args.eval_langs.split(",")) if lang.strip()]
        self.do_thinking_intervention = self.args.do_thinking_intervention

        # Here, we use every category dataset for calibration since the example is fairly limited if we only use reasoning categories.

        # Parallel English index: question_id -> {"question": str, "options_string": str}
        self.en_by_id: Dict[str, Dict[str, str]] = {}

        self.load_data()

    # -------------------- Utilities --------------------
    @staticmethod
    def _collect_options_from_datapoint(datapoint: Dict[str, Any]) -> List[str]:
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

    @staticmethod
    def _format_options_string(options: List[str]) -> str:
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


    def load_data(self):
        # Load test set with reasoning categories only

        if self.do_thinking_intervention:
            # 1) Build English index for parallel reference
            en_test = load_dataset("li-lab/MMLU-ProX-Lite", "en", split="validation")

            for dp in en_test:
                qid = dp["question_id"]
                en_question = dp["question"]
                en_options = self._collect_options_from_datapoint(dp)
                en_options_string = self._format_options_string(en_options)
                self.en_by_id[qid] = {
                    "question": en_question,
                    "options_string": en_options_string
                }

            # 2) Load and filter per evaluation language
            for lang in self.eval_langs:
                test_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="validation")

                self.datasets[lang] = test_dataset
                print(f"Loaded {len(test_dataset)} samples for language: {lang}")

        else:
            for lang in self.eval_langs:
                test_dataset = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="validation")

                self.datasets[lang] = test_dataset
                print(f"Loaded {len(test_dataset)} samples for language: {lang}")

    def get_prompt(self, question, category, options, lang):
        lang_lib_template = LANG_LIBS.get(lang, "")
        question_formatter = lang_lib_template[0]
        option_formatter = lang_lib_template[1]

        ans_suffix = lang_lib_template[5].format("X")
        subject_in_lang = ""
        if category == "computer science":
            # Use "computer_science" as the key if "computer science" is not found
            try:
                subject_in_lang = LANG_SUBJECTS[lang][category]
            except KeyError:
                subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
        else:
            subject_in_lang = LANG_SUBJECTS[lang][category]

        assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
        options_string = self._format_options_string(options)
        prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
        return prompt 

    def get_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        all_test_data = {}
        options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for lang, language_dataset in self.datasets.items():
            dataset_list = []
            for datapoint in language_dataset:
                id = datapoint["question_id"]
                question = datapoint["question"]
                answer = datapoint["answer"]

                options = self._collect_options_from_datapoint(datapoint)
                options_dict: Dict[str, str] = {}
                for idx, opt in enumerate(options):
                    options_dict[options_letters[idx]] = opt
                prompt = self.get_prompt(question, datapoint["category"], options, lang)


                if "gpt-oss" in self.args.model_name or "Qwen3" in self.args.model_name:
                    # By default, QwQ tokenizer will append "<think>" to the response
                    prompt_dict = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                    if self.do_thinking_intervention:
                        if self.args.thinking_intervention_lang == "en":
                            en_question = self.en_by_id[id]["question"]
                            en_options_string = self.en_by_id[id]["options_string"]
                            thinking_intervention_quote = f"Okay, let's see. I understand the question as: {en_question}', and options as: {en_options_string}. Let's solve the problem based on this understanding."
                        else:
                            raise ValueError(f"Unsupported thinking intervention language: {self.args.thinking_intervention_lang}")
                        if "gpt-oss" in self.args.model_name:
                            # Here, gpt-oss use "harmony" format which does not use "<think>"
                            prompt_dict.append({
                                "role": "assistant",
                                "content": thinking_intervention_quote
                            })
                        else:
                            prompt_dict.append({
                                "role": "assistant",
                                "content": "<think>\n" + thinking_intervention_quote
                            })
                else:
                    raise ValueError(f"Unsupported model for MMLU-ProX lite: {self.args.model_name}")
                dataset_list.append({
                    "id": id,
                    "language_code": lang,
                    "question": question,
                    "answer": answer,
                    "prompt_dict": prompt_dict,
                    "options_dict": options_dict
                })
            all_test_data[lang] = dataset_list

        return all_test_data



class UnderStandabilityEvalDataset:
    """
    Build dataset-like rows using both normal and thinking-intervention eval results.
    - Filter: keep samples where either normal or intv is correct (==1.0)
    - Label (understandable): True iff normal run is correct (==1.0)
    - Provide reasoning traces for both runs for token-length estimation.
    Output format: lang -> [ { id, understandable(int), reasoning_trace, reasoning_trace_intv, correct(int) }, ... ]
    Note: `correct` here is the normal run correctness for convenience.
    """

    def __init__(
        self,
        args,
        eval_langs: List[str]
    ) -> None:
        self.model_name = args.model_name
        self.eval_langs = eval_langs
        self.task_eval_results_path = args.task_eval_results_path
        self.thinking_intv_eval_results_path = args.thinking_intv_eval_results_path
        self.dataset_type = args.dataset_type
        # For PolyMath we expose split; for MMLU we map to validation/test
        self.polymath_split = args.polymath_split
        self.rows_by_lang: Dict[str, List[Dict[str, Any]]] = {}
        self.args = args
        self._load()

    def get_prompt_for_mmlu_prox(self, question, category, options, lang):
        lang_lib_template = LANG_LIBS.get(lang, "")
        question_formatter = lang_lib_template[0]
        option_formatter = lang_lib_template[1]

        ans_suffix = lang_lib_template[5].format("X")
        subject_in_lang = ""
        if category == "computer science":
            # Use "computer_science" as the key if "computer science" is not found
            try:
                subject_in_lang = LANG_SUBJECTS[lang][category]
            except KeyError:
                subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
        else:
            subject_in_lang = LANG_SUBJECTS[lang][category]

        assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
        options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        options_string = ""
        for idx, option in enumerate(options):
            options_string += f"({options_letters[idx]}) {option}\n"
        options_string = options_string.rstrip("\n")
        prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
        return prompt 


    def _load(self) -> None:
        model_base = self.model_name.split("/")[-1]
        assert model_base in self.task_eval_results_path, (
            f"Model name {model_base} not found in task eval path {self.task_eval_results_path}"
        )
        assert model_base in self.thinking_intv_eval_results_path, (
            f"Model name {model_base} not found in thinking intervention eval path {self.thinking_intv_eval_results_path}"
        )

        with open(self.task_eval_results_path, "r", encoding="utf-8") as f:
            self.task_eval_results = json.load(f)
        
        with open(self.thinking_intv_eval_results_path, "r", encoding="utf-8") as f:
            self.think_intv_eval_results = json.load(f)

        for lang in self.eval_langs:
            nmap = self.task_eval_results.get(lang, {}) or {}
            imap = self.think_intv_eval_results.get(lang, {}) or {}
            if not nmap and not imap:
                logger.warning("No entries for lang=%s in provided eval files", lang)
                continue

            # Load source dataset for question/answer lookup
            src_map: Dict[str, Dict[str, Any]] = {}
            split_name = None
            if self.dataset_type == "polymath":
                split_name = self.polymath_split
                if self.args.low_resource_experiment:
                    with open(self.args.translated_dataset_json_path, "r", encoding="utf-8") as f:
                        ds = json.load(f)
                        ds = ds.get(lang, [])
                    if lang == "fr":
                        # French dataset already present in the dataset, so we use that 
                        ds = load_dataset("Qwen/PolyMath", lang, split=split_name)
                else:
                    ds = load_dataset("Qwen/PolyMath", lang, split=split_name)
                for dp in ds:
                    sid = str(dp.get("id"))
                    if self.args.low_resource_experiment: 
                        sid = sid.replace("en", lang)
                    src_map[sid] = dp
            elif self.dataset_type in ("mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mmlu_prox_lite"):
                split_name = "validation" if self.dataset_type.endswith("dev") else "test"
                ds = load_dataset("li-lab/MMLU-ProX-Lite", lang, split=split_name)
                for dp in ds:
                    sid = str(dp.get("question_id"))
                    src_map[sid] = dp
            elif self.dataset_type == "mgsm_filtered":
                split_name = "filtered"
                ds = load_dataset("deokhk/filtered_mgsm_with_ids", split=lang)
                for idx, dp in enumerate(ds):
                    sid = str(idx)
                    src_map[sid] = dp
            else:
                raise ValueError(f"Unsupported dataset_type for understandability eval: {self.dataset_type}")

            rows: List[Dict[str, Any]] = []
            # Iterate by union of IDs present in both maps to filter; but require presence in both maps
            ids = set(nmap.keys()) & set(imap.keys())
            for sid in ids:
                ne = nmap.get(sid)
                ie = imap.get(sid)
                if not ne or not ie:
                    continue
                n_correct = int(ne.get("correct", 0))
                i_correct = int(ie.get("correct", 0))
                # Filter: keep if either is correct
                if (n_correct == 1) or (i_correct == 1):
                    sid_norm = str(sid)
                    src = src_map.get(sid_norm)
                    if not src:
                        logger.warning("Source dataset row not found for %s:%s, skipping", lang, sid_norm)
                        continue

                    # Extract question/answer by dataset_type
                    if self.dataset_type == "polymath":
                        if self.args.low_resource_experiment:
                            if lang == "fr":
                                question = src.get("question", "")
                            else:
                                question = src.get("translated_question", "")
                        else:
                            question = src.get("question", "")
                        answer = src.get("answer", "")
                        # Formatting instruction for math
                        try:
                            formatted_question = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                        except KeyError:
                            formatted_question = question
                        split_emit = self.polymath_split
                    elif self.dataset_type in ("mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mmlu_prox_lite"):

                        question = src.get("question", "")
                        answer = src.get("answer", "")
                        formatted_question = question  # no math boxing instruction here
                        split_emit = split_name or "test"
                        options_dict = {}
                        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
                        options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                        options = [src[field] for field in option_fields if field in src and (src[field] != None and src[field] != "N/A")]

                        for idx, option in enumerate(options):
                            options_dict[options_letters[idx]] = option
                        formatted_question = self.get_prompt_for_mmlu_prox(question, src["category"], options, lang)
                    else:  # mgsm_filtered
                        question = src.get("question", "")
                        # In filtered MGSM, gold is in 'answer_number'
                        answer = src.get("answer_number", src.get("answer", ""))
                        try:
                            formatted_question = question + "\n" + MATH_LANG_TO_FORMATTING_INSTRUCTION[lang]
                        except KeyError:
                            formatted_question = question
                        split_emit = split_name or "filtered"

                    task_reasoning_trace = ne.get("reasoning_trace", "")
                    task_prediction = str(ne.get("prediction", ""))
                    understandable = True if n_correct == 1 else False

                    # Build chat-like prompt content for compatibility
                    original_prompt_dict_input_and_reasoning_trace = [
                        {"role": "user", "content": formatted_question},
                        {"role": "assistant", "content": task_reasoning_trace},
                    ]

                    rows.append({
                        "id": sid_norm,
                        "language_code": lang,
                        "split": split_emit,
                        "question": question,
                        "formatted_question": formatted_question,
                        "answer": answer,
                        "reasoning_trace": task_reasoning_trace,
                        "prediction": task_prediction,
                        "understandable": understandable,
                        "original_prompt_dict_input_and_reasoning_trace": original_prompt_dict_input_and_reasoning_trace,
                    })
            if rows:
                self.rows_by_lang[lang] = rows
            logger.info(
                "Language %s: kept %d samples after filtering (either normal or intv correct)",
                lang, len(rows)
            )

    def get(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.rows_by_lang