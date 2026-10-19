"""RULER niah_single_2/3 with the text around the haystack changed: where do Gamma cues stop helping?

Haystack, needles, depths, answers, generation prefix and scoring are lm-eval's own
(`lm_eval.tasks.ruler`); only the prompt template differs:

- qend: no opening instruction; only the question at the end names the target.
- preamble: a preamble of about 90 tokens before RULER's instruction, longer than the
  64-token cue region (log_kv_gamma_cue_tokens), so the region ends inside the preamble.
- paraphrase: the opening instruction shares no rare word with the needle.

Gamma cues read the opening region, so all three are expected to fall back to the
no-cue score; RULER's own template is the case cues were built for.
"""
from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, aggregate_metrics, get_tokenizer, process_results  # noqa: F401
from lm_eval.tasks.ruler.niah_utils import TEMPLATE, download_dataset
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

# "{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
QUESTION = TEMPLATE[TEMPLATE.index("{context}"):]
PREAMBLE = (
    "You will read a long document taken from a collection of essays written by one author. Read the whole "
    "document from beginning to end, keep track of every fact it states, and then reply to the question that "
    "follows it. Every part of the document matters, so do not skip any of it. Give only the exact value that "
    "the question asks for. Do not explain your reasoning, do not add any commentary, and do not repeat the "
    "question. The document begins below this line.\n"
)
PARAPHRASE = ("A secret value is hidden within the following text. Make sure to memorize it. "
              "I will quiz you about it afterwards.\n")
TEMPLATES = dict(qend=QUESTION, preamble=PREAMBLE + TEMPLATE, paraphrase=PARAPHRASE + QUESTION)
# RULER rewrites these words in single-value templates ("Some" -> "A", "are" -> "is", ...); the added
# text must not contain them, and must not name the needle's words.
for _text in (PREAMBLE, PARAPHRASE):
    assert not any(word in _text for word in ("Some", "are", "answers", "special", "magic", "number", "uuid"))


def _samples(variant, type_needle_v, **kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer = get_tokenizer(**kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATES[variant],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v=type_needle_v,
            num_samples=500,
            TOKENIZER=tokenizer,
        )
        for seq in seq_lengths
    )


def niah_single_2_qend(**kwargs):
    return _samples("qend", "numbers", **kwargs)


def niah_single_2_preamble(**kwargs):
    return _samples("preamble", "numbers", **kwargs)


def niah_single_2_paraphrase(**kwargs):
    return _samples("paraphrase", "numbers", **kwargs)


def niah_single_3_qend(**kwargs):
    return _samples("qend", "uuids", **kwargs)


def niah_single_3_preamble(**kwargs):
    return _samples("preamble", "uuids", **kwargs)


def niah_single_3_paraphrase(**kwargs):
    return _samples("paraphrase", "uuids", **kwargs)
