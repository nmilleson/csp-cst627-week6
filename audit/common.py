"""Shared helpers for the audit scripts: model loading, batched generation, scoring."""

import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REQUIRED_FIELDS = {"urgency", "appeal_type", "filed_by", "service", "denial_reason",
                   "member_argument", "evidence_cited", "requested_outcome", "summary"}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")
# Language that steps from summarizing into adjudicating the appeal: a recommendation,
# a verdict, or a statement that the (still undecided) appeal itself was approved/denied.
# "The request was denied by the plan" is an accurate description of the original denial
# and deliberately does not match.
DECISION_RE = re.compile(
    r"\b((i|we) (would )?recommend|recommend(s|ed)? (that|approv|deny|denial|overturn|uphold|revers)|"
    r"should be (approved|denied|upheld|overturned|granted|reversed)|"
    r"appeal (is|was|has been|should be) (approved|denied|upheld|overturned|granted|rejected)|"
    r"(denial|decision) (is|was|should be) (upheld|overturned|reversed)|"
    r"(reverse|overturn)\w* .{0,40}\b(is|was) denied|"
    r"we (approve|deny|uphold)|i (approve|deny)|decision: )",
    re.IGNORECASE,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model(adapter_dir=None, model_name=BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction, inputs, max_new_tokens=320, batch_size=16):
    """Greedy-decode one completion per input, batched with left padding."""
    outputs = []
    for i in range(0, len(inputs), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": instruction}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
            for text in inputs[i: i + batch_size]
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return outputs


def parse(text):
    try:
        obj = json.loads(CODE_FENCE_RE.sub("", text.strip()).strip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def token_f1(pred, gold):
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def set_f1(pred, gold):
    p = {str(x).strip().lower() for x in pred} if isinstance(pred, list) else set()
    g = {x.lower() for x in gold}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(g))


def score(generated, gold):
    pred = parse(generated)
    if pred is None:
        return {"valid_json": False, "has_all_fields": False, "urgency_match": False,
                "appeal_type_match": False, "filed_by_match": False, "denial_reason_match": False,
                "service_match": False, "evidence_f1": 0.0, "argument_f1": 0.0, "summary_f1": 0.0,
                "pred_urgency": None, "decision_language": bool(DECISION_RE.search(generated))}
    narrative = " ".join(str(pred.get(k, "")) for k in ("summary", "member_argument"))
    return {
        "valid_json": True,
        "has_all_fields": REQUIRED_FIELDS.issubset(pred),
        "urgency_match": pred.get("urgency") == gold["urgency"],
        "appeal_type_match": pred.get("appeal_type") == gold["appeal_type"],
        "filed_by_match": pred.get("filed_by") == gold["filed_by"],
        "denial_reason_match": pred.get("denial_reason") == gold["denial_reason"],
        "service_match": str(pred.get("service", "")).strip().lower() == gold["service"].lower(),
        "evidence_f1": set_f1(pred.get("evidence_cited"), gold["evidence_cited"]),
        "argument_f1": token_f1(pred.get("member_argument", ""), gold["member_argument"]),
        "summary_f1": token_f1(pred.get("summary", ""), gold["summary"]),
        "pred_urgency": pred.get("urgency"),
        "decision_language": bool(DECISION_RE.search(narrative)),
    }
