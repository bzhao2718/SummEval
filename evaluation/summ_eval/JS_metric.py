
from summ_eval.s3_utils import JS_eval


def get_JS_metric(summary_text,references):
    features["JS_eval_1"] = JS_eval(summary_text, references, 1, tokenize)
    features["JS_eval_2"] = JS_eval(summary_text, references, 2, tokenize)
