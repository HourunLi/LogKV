"""InfiniteBench (Zhang et al., 2024, "∞Bench: Extending Long Context Evaluation Beyond 100K Tokens") at a fixed
context budget: infinitebench_<task>_<64k|128k|256k>, grouped as infinitebench_<length>.

The ten tasks that KV-cache papers report (MInference, LAVa, ReST-KV ...): Retr.PassKey, Retr.Number, Retr.KV,
En.Sum, En.QA, En.MC, Zh.QA, En.Dia, Code.Debug, Math.Find. Code.Run and Math.Calc, near zero for every model in
the paper, are left out.

As the official code (OpenBMB/InfiniteBench, src/eval_yarn_mistral.py, src/compute_scores.py):
- prompts: the templates the paper uses for base models (yarn_mistral_templates; the model has no chat template);
- the whole prompt is cut in the middle to the budget, keeping its first and last halves (the instruction at the
  start, the question at the end), where the official code cuts to 128K;
- answers are greedy, with the official max_new_tokens, and scored by the official rules (first integer, word
  match, option letter, token F1, character F1 for Chinese, ROUGE-Lsum for the summary).
The budget is the length, less what the generation needs beyond the window's 1024-token margin (En.Sum's 1200),
so eval.py never cuts the prompt from its start. Contexts shorter than the budget (most samples at 256K; the
retrieval contexts are about 128K tokens) are kept whole.

Data: the jsonl files of the dataset xinrongzhang2022/InfiniteBench, from INFINITEBENCH_DIR (a folder holding them)
or the hub cache (unused/fetch_long_eval_data.py --infinitebench downloads them).
"""
from __future__ import annotations

import functools
import json
import os
import re
import string
from collections import Counter

import datasets

REPO = "xinrongzhang2022/InfiniteBench"
MARGIN = 1024  # the window beyond the length (long_eval_plan.MARGIN)
SLACK = 64  # tokens that decoding the cut prompt and encoding it again may add

# task: (prompt template, max_new_tokens), as yarn_mistral_templates and DATA_NAME_TO_MAX_NEW_TOKENS.
TASKS = {
    "passkey": ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will "
                "quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is", 6),
    "number_string": ("There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you "
                      "about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
                      12),
    "kv_retrieval": ("Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n"
                     "{input}", 50),
    "longbook_sum_eng": ("Summarize the book below.\n\n{context}\n\nSummary:", 1200),
    "longbook_qa_eng": ("Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\n"
                        "Question: {question}\nAnswer:", 40),
    "longbook_choice_eng": ("Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n"
                            "A. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\n"
                            "The letter of the correct answer is", 40),
    "longbook_qa_chn": ("阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：", 40),
    "longdialogue_qa_eng": ("Below is a dialogue script where one random occurrence of a character name is replaced "
                            "with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\n"
                            "The name that has been replaced with $$MASK$$ is likely", 40),
    "code_debug": ("Following is a Python code where exactly one of the functions/methods has a deliberate error that "
                   "makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\n"
                   "D. {OPTION_D}\n\nThe correct option is:", 5),
    "math_find": ("{prefix}\n\n{context}\n\n{input}", 3),
}
LENGTHS = {"64k": 65536, "128k": 131072, "256k": 262144}


def data_file(task: str) -> str:
    folder = os.environ.get("INFINITEBENCH_DIR")
    if folder:
        return os.path.join(folder, f"{task}.jsonl")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, f"{task}.jsonl", repo_type="dataset")


def prompt_of(task: str, eg: dict) -> str:
    """create_prompt of the official code, for its yarn-mistral templates."""
    template = TASKS[task][0]
    options = eg.get("options") or []
    letters = {f"OPTION_{c}": (options[i] if i < len(options) else "") for i, c in enumerate("ABCD")}
    if task == "math_find":
        target = re.findall(r"The .+ of", eg["input"])
        assert target, f"Cannot find the target number in {eg['input']}"
        prefix = f"What is {target[0].lower()[:-3]} in the following list?"
        return template.format(prefix=prefix, context=eg["context"], input=eg["input"])
    return template.format(context=eg["context"], input=eg["input"], question=eg["input"], **letters)


def answer_of(task: str, eg: dict):
    """get_answer of the official code: for the choice tasks [option text, letter]."""
    answer = eg["answer"]
    if task in ("code_debug", "longbook_choice_eng"):
        if isinstance(answer, list) and len(answer) == 2 and answer[1] in ("A", "B", "C", "D"):
            return answer
        text = answer[0] if isinstance(answer, list) else answer
        options = eg.get("options") or []
        # An answer that is none of the options (the official code would stop) gets no letter that can match.
        return [text, "ABCD"[options.index(text)] if text in options[:4] else "<none>"]
    return answer


def _rows(task, path, length, tokenizer):
    """One small row per sample: where its line starts in the file and its answer; doc_to_text reads the line."""
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            if line.strip():
                eg = json.loads(line)
                yield dict(id=str(eg.get("id", "")), task=task, length=length, tokenizer=tokenizer, path=path,
                           offset=offset, answer=json.dumps(answer_of(task, eg), ensure_ascii=False))
            offset += len(line)


def load(**kwargs):
    """custom_dataset (dataset_kwargs infinitebench_task / infinitebench_length; the tokenizer from the metadata).
    The samples stay in the data file (a retrieval file holds about a gigabyte of text): a row keeps the offset
    of its line, and doc_to_text reads, prompts and cuts only the samples that run."""
    task, length = kwargs["infinitebench_task"], kwargs["infinitebench_length"]
    tokenizer = kwargs.get("tokenizer") or kwargs.get("pretrained")
    if not tokenizer:
        raise ValueError("infinitebench needs the tokenizer: metadata pretrained (eval.py passes it)")
    return {"test": datasets.Dataset.from_list(list(_rows(task, data_file(task), length, tokenizer)))}


def sample(doc: dict) -> dict:
    with open(doc["path"], "rb") as f:
        f.seek(doc["offset"])
        return json.loads(f.readline())


@functools.cache
def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def budget(task: str, length: str) -> int:
    n = LENGTHS[length]
    return min(n, n + MARGIN - TASKS[task][1] - SLACK)


def truncate_middle(tokens: list, max_length: int) -> list:
    """truncate_input(manner="middle") of the official code."""
    if len(tokens) <= max_length:
        return tokens
    return tokens[:max_length // 2] + tokens[-max_length // 2:]


def doc_to_text(doc: dict) -> str:
    prompt = prompt_of(doc["task"], sample(doc))
    tok = _tokenizer(doc["tokenizer"])
    tokens = tok(prompt, add_special_tokens=False)["input_ids"]
    limit = budget(doc["task"], doc["length"])
    if len(tokens) <= limit:
        return prompt
    return tok.decode(truncate_middle(tokens, limit), skip_special_tokens=True)


def doc_to_target(doc: dict) -> str:
    answer = json.loads(doc["answer"])
    return str(answer[0] if isinstance(answer, list) else answer)


# --- scoring: src/compute_scores.py --------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def normalize_zh_answer(s: str) -> str:
    cn_punctuation = ("！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟"
                      "〰〾〿–—‘’‛“”„‟…‧﹏.")
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join("".join(ch for ch in s.lower() if ch not in punctuation).split())


def f1(prediction: list, truth: list) -> float:
    same = sum((Counter(prediction) & Counter(truth)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(prediction), same / len(truth)
    return 2 * precision * recall / (precision + recall)


def first_int_match(prediction: str) -> str:
    return next((item for item in re.split("[^0-9]", prediction) if item), "")


def _first(label):
    return label[0] if isinstance(label, list) else label


def score_kv_retrieval(pred, label):
    for c in ['\n', ':', '"', "'", '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    return float(_first(label) in pred.split())


def score_passkey(pred, label):
    return float(str(_first(label)) == first_int_match(pred))


def score_code_debug(pred, label):
    pred = pred.strip()
    fn_name, label_c = label[0], label[1]
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", pred)
    if match and match.group(0) == label_c:
        return 1.0
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return 1.0
    for prefix in ("answer is:", "is:", "answer:", "correct option is:"):
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return 0.0
        rest = pred[idx + len(prefix) + 1:]
        return float(rest.startswith(label_c) or rest.startswith(fn_name))
    return 0.0


def score_math_find(pred, label):
    label = _first(label)
    number = re.search(r"\d+\.\d+|\d+", pred)
    if number is None:
        return 0.0
    if isinstance(label, int):
        return float(int(float(number.group(0))) == label) if "." not in number.group(0) else 0.0
    return float(float(number.group(0)) == label)


def score_dialogue(pred, label):
    pred = pred.strip().upper()
    return float(any(item.upper() in pred for item in label))


def score_choice(pred, label):
    pred = pred.strip()
    match = re.search(r"\b[A-D]\b(?!.*\b[A-D]\b)", pred)
    if match and match.group(0) in label:
        return 1.0
    if pred == "":
        return 0.0
    if pred[0] in "ABCD":
        return float(pred[0] in label)
    if pred in label:
        return 1.0
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    for prefix in ("answer is:", "answer:", "answer is", "option is"):
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return 0.0
        rest = pred[idx + len(prefix) + 1:]
        return float(any(rest.startswith(s) for s in label))
    for word in pred.split():
        if word in "ABCD":
            return float(word in label)
    return 0.0


def score_qa_eng(pred, label):
    labels = label if isinstance(label, list) else [label]
    return max(f1(normalize_answer(pred).split(), normalize_answer(t).split()) for t in labels)


def score_qa_chn(pred, label):
    labels = label if isinstance(label, list) else [label]
    return max(f1(list(normalize_zh_answer(pred)), list(normalize_zh_answer(t))) for t in labels)


@functools.cache
def _rouge():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeLsum"])  # as evaluate's "rouge": no stemmer, sentences on newlines


def score_sum(pred, label):
    labels = label if isinstance(label, list) else [label]
    return max(_rouge().score(t, pred)["rougeLsum"].fmeasure for t in labels)


SCORERS = {
    "passkey": score_passkey, "number_string": score_passkey, "kv_retrieval": score_kv_retrieval,
    "longbook_sum_eng": score_sum, "longbook_qa_eng": score_qa_eng, "longbook_choice_eng": score_choice,
    "longbook_qa_chn": score_qa_chn, "longdialogue_qa_eng": score_dialogue, "code_debug": score_code_debug,
    "math_find": score_math_find,
}


def process_results(doc, results):
    return {"score": SCORERS[doc["task"]](results[0], json.loads(doc["answer"]))}
