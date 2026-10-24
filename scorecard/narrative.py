"""Short written explanations of a decision for the loan officer and the customer, with automated checks.

The LLM only rewords the reason codes into a few sentences. Every draft goes through these checks:
  decision      states the decision that was made, and no other
  numbers       every number in the text appears in the reason codes (or the PD, for the officer), and the
                numbers in one sentence all come from the same reason, so a figure can't move to another factor
  factors       no risk factor that isn't in the reason codes given to that reader
  protected     no mention or guess of gender, marital status, race, religion or nationality
  officer_only  (customer only) nothing about age, location, contact details or documents
  coverage      every reason's value appears in the text, so the reasons aren't blurred into vague summaries
  judgement     no judgement words of the model's own (significant, poor, below average, ...)
  format        plain prose: no list markers and no prompt text or feedback echoed back
The opening sentence that states the decision is added by the code, not written by the model, so the
decision check is really checking that the rest of the text doesn't contradict it.
A draft that fails any check is replaced by a plain template built from the reason codes, and the
result records that it fell back. By default only the officer gets LLM text; the customer gets the template.

The customer prompt is only ever given the customer reasons, so officer-only facts can't leak through
the prompt; the officer_only check is there in case the model invents them.
"""

from __future__ import annotations

import re

from .llm import LLM

DECISION_WORDS = {
    "officer": {"approve": "approve", "review": "manual review", "decline": "decline"},
    "customer": {"approve": "approved", "review": "passed to a loan officer for review", "decline": "declined"},
}
# labels shown above the explanation in the app and the API, so nobody mistakes a draft for a decision letter
LLM_LABEL = "AI-drafted summary: check against the reason codes below"
FALLBACK_LABEL = "Template summary (the AI draft failed an automated check)"
TEMPLATE_LABEL = "Template summary built from the reason codes"

# the first sentence is added by the code, so the decision is always stated the same way
OPENING = {
    "officer": "Recommendation: {words}.",
    "customer": "Your application has been {words}.",
}
# stems that signal each decision; a text must use its own and none of the others
DECISION_PATTERNS = {
    "approve": re.compile(r"\bapprov\w*", re.I),
    "review": re.compile(r"\breview\w*|\brefer\w*", re.I),
    "decline": re.compile(r"\bdeclin\w*", re.I),
}

PROTECTED = re.compile(
    r"\b(gender|sex|male|female|man|woman|men|women|he|she|his|her|hers|him|himself|herself|husband|wife|spouse|"
    r"married|marriage|marital|single|divorc\w*|widow\w*|separated|pregnan\w*|race|racial|ethnic\w*|religio\w*|"
    r"nationality)\b", re.I)

# age as a personal attribute ("age of your car" or "age of loans" is fine), plus contact and document details
OFFICER_ONLY = {
    "age": re.compile(r"\bage[ds]?\b(?!\s+of\s+(?:the\s+|your\s+)?(?:car|vehicle|loans?|building|previous|applications?))"
                      r"|\byears old\b|\belderly\b|\bretire\w*|\bpension\w*|\bsenior\b|\bbirth\w*|\byoung\w*|\bolder\b", re.I),
    "contact details": re.compile(r"\bphone\w*|\bmobile\b|\be-?mail\w*", re.I),
    "documents": re.compile(r"\bdocument\w*|\bidentity\b", re.I),
    "ID": re.compile(r"\bID\b"),
    "social circle": re.compile(r"\bsocial circle\b|\bacquaint\w*", re.I),
    "registration": re.compile(r"\bregist\w*", re.I),
    "application time": re.compile(r"\bweekday\b|\bhour\b", re.I),
    "location": re.compile(r"\bregion\w*|\bcity\b|\bcities\b|\baddress\w*|\bpopulation\b|\bneighbourhood\b", re.I),
    "accompanied": re.compile(r"\baccompan\w*", re.I),
}

# list markers or bits of the prompt copied into the answer
SCAFFOLD = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+|\bReasons \(each|\bOpening sentence\b|^\s*Decision:|"
                      r"\bprevious draft\b|\bbroke these rules\b|\bcorrected version\b|\"\"\"", re.I | re.M)

# the model's own verdicts; the reason codes give the value and the comparison, and that's all the text should say
JUDGEMENT = re.compile(r"\b(significant\w*|substantial\w*|extreme\w*|poor\w*|weak\w*|bad|good|excellent|strong\w*|"
                       r"decreas\w*|neither|concern\w*|worr\w*|unusual\w*|alarming|below average|above average)\b", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# risk factors the model might bring up on its own; any of these must also appear in the reasons it was given
FACTOR_TERMS = {
    "income": r"\bincome\w*|\bsalar\w*|\bearn\w*",
    "age": r"\bage[ds]?\b",
    "employment": r"\bemploy\w*|\bjobs?\b|\boccupation\w*",
    "education": r"\beducat\w*|\bdegree\b",
    "household": r"\bchild\w*|\bhousehold\b|\bfamil\w*|\bdependants?\b",
    "car": r"\bcars?\b|\bvehicle\w*",
    "housing": r"\bpropert\w*|\bhous(e|es|ing)\b|\bapartment\w*|\brealty\b",
    "region": r"\bregion\w*|\bcity\b|\bcities\b",
    "bureau": r"\bbureau\w*",
    "score": r"\bscores?\b",
    "late payments": r"\boverdue\b|\barrear\w*|\blate\b|\blateness\b|\bmissed\b|\bwritten off\b",
    "debt": r"\bdebts?\b|\bowed?\b|\bowing\b|\boutstanding\b",
    "refusals": r"\brefus\w*|\breject\w*",
    "instalments": r"\binstal\w*",
    "term": r"\bterms?\b",
    "goods price": r"\bgoods\b",
    "credit cards": r"\bcards?\b",
    "credit limit": r"\blimits?\b",
    "contact details": r"\bphone\w*|\bmobile\b|\be-?mail\w*",
    "documents": r"\bdocument\w*",
    "social circle": r"\bsocial circle\b",
}
FACTOR_TERMS = {k: re.compile(v, re.I) for k, v in FACTOR_TERMS.items()}

# a number with optional thousands separators and decimals; list markers at the start of a line are removed first
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.M)

SYSTEM = {
    "officer": (
        "You write short explanations of credit decisions for a loan officer at a consumer lender. "
        "Follow every rule:\n"
        "1. Mention only the factors listed under Reasons. Do not add any other factor, cause, advice or assumption.\n"
        "2. Copy every number exactly as written in the input. Do not round, convert or work out new numbers.\n"
        "3. Do not state or hint at a decision; the opening sentence does that. Do not use the words approve, review "
        "or decline.\n"
        "4. Mention every reason with its value exactly as given and its comparison with other applicants.\n"
        "5. Do not add your own judgement such as good, poor, weak or below average; the comparison says enough.\n"
        "6. Never mention or guess gender, marital status, race, religion or nationality. Do not use he, she, his "
        "or her; write 'the applicant'.\n"
        "7. Write one sentence per reason, as plain prose. No lists, headings, greetings or sign-off.\n\n"
        "Example. Reasons: 'Credit card balance as a share of the limit: 92.0% (highest 10% of applicants)'; "
        "'Years in current job: 0.8 years (lowest 20% of applicants)'. Answer: 'The applicant's credit card balance "
        "is 92.0% of the limit, in the highest 10% of applicants. The applicant has been in the current job for 0.8 "
        "years, in the lowest 20% of applicants.'"
    ),
    "customer": (
        "You write short, neutral explanations of a credit decision for the applicant, in plain English. "
        "Follow every rule:\n"
        "1. Mention only the factors listed under Reasons. Do not add any other factor, cause, advice or assumption.\n"
        "2. Copy every number exactly as written in the input. Do not round, convert or work out new numbers.\n"
        "3. Do not state or hint at a decision; the opening sentence does that. Do not use the words approve, review "
        "or decline.\n"
        "4. Mention every reason with its value exactly as given and its comparison with other applicants.\n"
        "5. Do not add your own judgement such as good, poor, weak or below average; the comparison says enough.\n"
        "6. Never mention or guess the applicant's age, gender, marital status, race, religion or nationality, or "
        "their contact details or documents.\n"
        "7. Address the applicant as 'you'. Be polite and factual, without blame or apology.\n"
        "8. Write one sentence per reason, as plain prose. No lists, headings, greetings or sign-off.\n\n"
        "Example. Reasons: 'Credit card balance as a share of the limit: 92.0% (highest 10% of applicants)'; "
        "'Years in current job: 0.8 years (lowest 20% of applicants)'. Answer: 'Your credit card balance is 92.0% of "
        "the limit, which is in the highest 10% of applicants. You have been in your current job for 0.8 years, which "
        "is in the lowest 20% of applicants.'"
    ),
}


def payload_from_explanation(ex: dict, audience: str) -> dict:
    """What one reader's explanation is built from: the decision, PD band and that reader's reason codes."""
    reasons = ex["reasons_officer"] if audience == "officer" else ex["reasons_customer"]
    return {
        "audience": audience,
        "decision": ex["decision"],
        "pd": ex["pd"],
        "band": ex["risk_band"]["band"],
        "band_label": ex["risk_band"]["label"],
        "reasons": [r["text"] for r in reasons],
    }


def _pd_line(payload: dict) -> str:
    return f"PD: {payload['pd']:.1%} (risk band {payload['band']}, {payload['band_label']})"


def allowed_text(payload: dict) -> str:
    # everything the reader's explanation may draw facts from; the customer doesn't get the PD
    lines = list(payload["reasons"])
    if payload["audience"] == "officer":
        lines.append(_pd_line(payload))
    return "\n".join(lines)


def build_prompt(payload: dict) -> tuple[str, str]:
    audience = payload["audience"]
    lines = [f"Decision: {payload['decision']}",
             f"The opening sentence \"{opening(payload)}\" is added for you. Write only the sentences that follow it."]
    if audience == "officer":
        lines.append(_pd_line(payload))
    lines.append("Reasons (each one increased the assessed risk, strongest first):")
    lines += [f"- {r}" for r in payload["reasons"]] or ["- none"]
    lines.append("Write the explanation now.")
    return SYSTEM[audience], "\n".join(lines)


def opening(payload: dict) -> str:
    return OPENING[payload["audience"]].format(words=DECISION_WORDS[payload["audience"]][payload["decision"]])


def reason_value(reason: str) -> str:
    # "Label: value (comparison)" gives "value"; a label can contain brackets but not ": "
    value = reason.split(": ", 1)[1] if ": " in reason else reason
    return value.rsplit(" (", 1)[0]


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in NUMBER.findall(LIST_MARKER.sub("", text))}


def run_checks(text: str, payload: dict) -> dict[str, dict]:
    """Every check for one draft, as {name: {"passed": bool, "detail": str}}."""
    allowed = allowed_text(payload)
    out = {}

    # decision: its own stem present, the other decisions' stems absent
    own = payload["decision"]
    others = [d for d, pat in DECISION_PATTERNS.items() if d != own and pat.search(text)]
    has_own = bool(DECISION_PATTERNS[own].search(text))
    out["decision"] = {"passed": has_own and not others,
                       "detail": "" if has_own and not others else
                       f"expected '{own}'" + ("" if has_own else " (missing)") + (f", also says {others}" if others else "")}

    # numbers: each one must come from the input, and each sentence's numbers from a single reason
    extra = sorted(_numbers(text) - _numbers(allowed))
    sources = [_numbers(r) for r in payload["reasons"]]
    if payload["audience"] == "officer":
        sources.append(_numbers(_pd_line(payload)))
    misplaced = [s for s in SENTENCE_END.split(text) if _numbers(s) and not any(_numbers(s) <= src for src in sources)]
    problems = ([f"numbers not in the reason codes: {extra}"] if extra else []) + \
               ([f"numbers mixed from different reasons in: {misplaced[0]!r}"] if misplaced and not extra else [])
    out["numbers"] = {"passed": not problems, "detail": "; ".join(problems)}

    # factors: terms for risk factors that the reasons given don't mention
    off_list = [name for name, pat in FACTOR_TERMS.items() if pat.search(text) and not pat.search(allowed)]
    out["factors"] = {"passed": not off_list, "detail": f"factors not in the reason codes: {off_list}" if off_list else ""}

    # protected and officer-only terms count only when the model added them: a reason code's own value
    # (e.g. employer sector "Religion") is allowed through, since the config decides what each reader sees
    hits = sorted({m.group(0).lower() for m in PROTECTED.finditer(text)}
                  - {m.group(0).lower() for m in PROTECTED.finditer(allowed)})
    out["protected"] = {"passed": not hits, "detail": f"protected terms: {hits}" if hits else ""}

    if payload["audience"] == "customer":
        # a term group is allowed when the reasons given to the customer use it themselves
        found = [name for name, pat in OFFICER_ONLY.items() if pat.search(text) and not pat.search(allowed)]
        out["officer_only"] = {"passed": not found, "detail": f"officer-only terms: {found}" if found else ""}

    # coverage: each reason's value shows up, as its numbers or, for a category, its text (Yes/No flags are skipped)
    missing = []
    text_numbers = _numbers(text)
    for reason in payload["reasons"]:
        value = reason_value(reason)
        nums = _numbers(value)
        if nums:
            covered = nums <= text_numbers
        else:
            covered = value in ("Yes", "No") or value.lower() in text.lower()
        if not covered:
            missing.append(value)
    out["coverage"] = {"passed": not missing, "detail": f"reason values missing: {missing}" if missing else ""}

    words = sorted({m.group(0).lower() for m in JUDGEMENT.finditer(text)}
                   - {m.group(0).lower() for m in JUDGEMENT.finditer(allowed)})
    out["judgement"] = {"passed": not words, "detail": f"judgement words: {words}" if words else ""}

    scaffold = SCAFFOLD.search(text)
    out["format"] = {"passed": not scaffold, "detail": f"not plain prose: {scaffold.group(0).strip()!r}" if scaffold else ""}
    return out


def template(payload: dict) -> str:
    """The plain fallback: the decision and the reason codes as written, so it passes every check by construction."""
    # one sentence per reason, so each sentence's numbers come from one reason
    reasons = " ".join(f"{r}." for r in payload["reasons"])
    if payload["audience"] == "officer":
        head = f"{opening(payload)} {_pd_line(payload)}."
        tail = f" The factors that raised the risk most, in order: {reasons}" if reasons else " No factor raised the risk."
        return head + tail
    head = opening(payload)
    tail = f" The factors that weighed most in this assessment were: {reasons}" if reasons else ""
    return head + tail


def _clean(text: str) -> str:
    # small models sometimes wrap the answer in quotes
    return text.strip().strip('"').strip()


def _failures(checks: dict) -> list[str]:
    return [f"{name}: {c['detail']}" for name, c in checks.items() if not c["passed"]]


def write_explanation(llm: LLM | None, payload: dict) -> dict:
    """One reader's explanation: a checked LLM draft, or the template if the draft fails any check.

    There is no retry. In the development runs a retry with the failures quoted back fixed none of them,
    and the model often echoed the feedback prompt. Returns the final text, its source ("llm",
    "template_fallback" or "template"), a display label, and the draft with its checks so the eval can
    count pass and fallback rates.
    """
    if llm is None:
        return {"text": template(payload), "source": "template", "label": TEMPLATE_LABEL, "attempts": [],
                "fell_back": False}
    system, user = build_prompt(payload)
    try:
        res = llm.generate(system, user)
    except Exception as exc:  # noqa: BLE001
        # a backend error falls back to the template rather than failing the scoring request
        attempt = {"text": "", "checks": {}, "error": str(exc), "latency_s": 0.0}
    else:
        body = _clean(res.text)
        # the model sometimes repeats the opening anyway, so I drop it before adding mine
        if body.startswith(opening(payload)):
            body = body[len(opening(payload)):].strip()
        text = f"{opening(payload)} {body}"
        attempt = {"text": text, "checks": run_checks(text, payload), "latency_s": res.latency_s}
        if not _failures(attempt["checks"]):
            return {"text": text, "source": "llm", "label": LLM_LABEL, "attempts": [attempt], "fell_back": False}
    return {"text": template(payload), "source": "template_fallback", "label": FALLBACK_LABEL, "attempts": [attempt],
            "fell_back": True}


def explain_both(llm: LLM | None, ex: dict, customer_llm: bool = False) -> dict:
    """Officer and customer explanations for one Explainer.explain result.

    The officer gets the checked LLM draft. The customer gets the template unless customer_llm is set
    (CUSTOMER_EXPLANATION=llm), because in the eval about 1 in 6 accepted customer drafts misread a reason.
    """
    return {
        "officer": write_explanation(llm, payload_from_explanation(ex, "officer")),
        "customer": write_explanation(llm if customer_llm else None, payload_from_explanation(ex, "customer")),
    }
