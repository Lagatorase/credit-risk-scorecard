"""Tests for the LLM explanation layer: the checks, the template fallback, and the backends (no real calls)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from scorecard.config import get_settings
from scorecard.llm import AnthropicLLM, LLMResult, OllamaLLM, OpenAILLM, get_llm
from scorecard.narrative import build_prompt, explain_both, run_checks, template, write_explanation
from scorecard.reasons import ReasonCoder
from tests.test_reasons import CONFIG, MODEL_FEATURES, _reference_for_all_features


def _payload(audience="officer", decision="decline", reasons=None):
    return {
        "audience": audience, "decision": decision, "pd": 0.293, "band": "E", "band_label": "very high",
        "reasons": reasons if reasons is not None else [
            "Average of the external bureau scores: 0.30 (lowest 10% of applicants)",
            "Largest overdue amount ever at other lenders: 582,898 (highest 10% of applicants)",
        ],
    }


def _failed(text, payload):
    return {name for name, c in run_checks(text, payload).items() if not c["passed"]}


class ScriptedLLM:
    """Returns the given replies in order, so I can test the fallback without a model."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, system, user, temperature=0.0, max_tokens=400):
        self.prompts.append(user)
        return LLMResult(self.replies.pop(0), self.name, 0.01)


GOOD = ("Recommendation: decline. The applicant's average external bureau score is 0.30, in the lowest 10% of "
        "applicants. Their largest overdue amount at other lenders was 582,898.")


def test_a_faithful_draft_passes_every_check():
    assert _failed(GOOD, _payload()) == set()
    # the officer may quote the PD exactly
    assert _failed(GOOD + " The PD is 29.3%.", _payload()) == set()


def test_each_check_catches_its_problem():
    p = _payload()
    assert _failed(GOOD.replace("decline", "approve"), p) == {"decision"}
    assert _failed(GOOD + " A manual review is not needed.", p) == {"decision"}
    # a rounded number and a made-up one both fail
    # (and the real value is then missing, so coverage fails too)
    assert _failed(GOOD.replace("582,898", "583,000"), p) == {"numbers", "coverage"}
    assert _failed(GOOD.replace("0.30", "0.3"), p) == set()  # same value, written differently
    assert _failed(GOOD + " The PD is 29%.", p) == {"numbers"}
    assert _failed(GOOD + " Their income is also low.", p) == {"factors"}
    assert _failed(GOOD + " This is a significant concern.", p) == {"judgement"}
    # each figure has to stay with its own reason
    swapped = ("Recommendation: decline. The applicant's average external bureau score is 0.30, in the highest 10% of "
               "applicants. Their largest overdue amount at other lenders was 582,898, in the lowest 10% of applicants.")
    assert _failed(swapped, p) == set()  # both 10%, so nothing moved
    moved = ("Recommendation: decline. The applicant's average external bureau score is 0.30, in the lowest 10% of "
             "applicants. Their largest overdue amount at other lenders was 582,898, in the lowest 20% of applicants.")
    assert _failed(moved, _payload(reasons=["Average of the external bureau scores: 0.30 (lowest 10% of applicants)",
                                            "Largest overdue amount ever at other lenders: 582,898 (highest 10% of applicants)",
                                            "Average yearly repayment on previous loans: 5,945 (lowest 20% of applicants)"])) \
        == {"numbers", "coverage"}  # the 20% belongs to the third reason, which the text leaves out
    # a vague summary that drops the values fails coverage
    assert _failed("Recommendation: decline. The applicant's bureau score and overdue history are weak.", p) == {"coverage", "judgement"}
    assert _failed(GOOD.replace("The applicant's", "His"), p) == {"protected"}


def test_customer_checks_cover_officer_only_details():
    p = _payload("customer", "review", reasons=["Credit bureau score: 0.40 (lowest 30% of applicants)"])
    ok = "Your application has been passed to a loan officer for review. Your credit bureau score is 0.40, in the lowest 30% of applicants."
    assert _failed(ok, p) == set()
    assert "officer_only" in _failed(ok + " Your age was also considered.", p)
    assert "officer_only" in _failed(ok + " You did not give a work phone.", p)
    assert "officer_only" in _failed(ok + " Your home region was also considered.", p)
    # the customer never gets the PD, so quoting it is an unsupported number
    assert _failed(ok + " Your PD is 29.3%.", p) == {"numbers"}
    # "age of your car" is about the car, and is fine when it's one of the reasons
    car = _payload("customer", "review", reasons=["Age of car: 12.0 years (highest 10% of applicants)"])
    text = "Your application has been passed to a loan officer for review. The age of your car, 12.0 years, was a factor."
    assert _failed(text, car) == set()


def test_format_check_catches_lists_and_echoed_prompts():
    p = _payload()
    # an echoed feedback prompt holds the old draft, so it can pass the other checks; format must catch it
    echo = 'Recommendation: decline. Your previous draft was: """' + GOOD.split(". ", 1)[1] + '""" Corrected version: ' + GOOD.split(". ", 1)[1]
    assert "format" in _failed(echo, p)
    echoed = "Recommendation: decline. Reasons (each one increased the assessed risk): " + GOOD.split(". ", 1)[1]
    assert _failed(echoed, p) == {"format"}
    assert _failed("Recommendation: decline.\n- " + GOOD.split(". ", 1)[1], p) == {"format"}


def test_customer_may_quote_a_term_from_its_own_reasons():
    # the officer-only check counts terms the model added; a term inside a reason given to the customer is
    # fine to repeat (location reasons are officer-only now, so this reason is made up for the test)
    p = _payload("customer", "approve", reasons=["Registered address in a different city from home: Yes (7.8% of applicants)"])
    assert _failed("Your application has been approved. Your registered address is in a different city from home, "
                   "as for 7.8% of applicants.", p) == set()


def test_reason_code_values_are_not_flagged_as_added_terms():
    # Home Credit has an employer sector called "Religion"; quoting the reason is not the model guessing
    p = _payload(reasons=["Employer sector: Religion (0.0% of applicants)"])
    assert _failed("Recommendation: decline. The applicant's employer sector is Religion.", p) == set()
    assert _failed("Recommendation: decline. The applicant's religion was a factor. She works there.", p) == {"protected"}


@pytest.mark.parametrize("seed", range(20))
def test_templates_pass_every_check(seed):
    # random reason lists over the real config, for every decision and both readers
    X, ref = _reference_for_all_features()
    coder = ReasonCoder(ref, CONFIG)
    shap_row = np.random.default_rng(seed).normal(size=len(MODEL_FEATURES))
    ex = {"pd": 0.17, "risk_band": {"band": "D", "label": "high"},
          "reasons_officer": coder.top(shap_row, X.iloc[seed], MODEL_FEATURES, audience="officer"),
          "reasons_customer": coder.top(shap_row, X.iloc[seed], MODEL_FEATURES, audience="customer")}
    for decision in ("approve", "review", "decline"):
        out = explain_both(None, {**ex, "decision": decision})
        for audience in ("officer", "customer"):
            assert out[audience]["source"] == "template"
            payload = {**_payload(audience, decision), "pd": 0.17, "band": "D", "band_label": "high",
                       "reasons": [r["text"] for r in ex[f"reasons_{audience}"]]}
            assert _failed(out[audience]["text"], payload) == set(), out[audience]["text"]


def test_first_draft_passes():
    out = write_explanation(ScriptedLLM([GOOD]), _payload())
    assert out["source"] == "llm" and not out["fell_back"] and out["text"] == GOOD
    assert out["label"].startswith("AI-drafted summary")


def test_the_opening_sentence_is_added_by_the_code():
    body = GOOD.split(". ", 1)[1]
    out = write_explanation(ScriptedLLM([body]), _payload())
    assert out["text"] == GOOD and out["source"] == "llm"


def test_a_failed_draft_goes_straight_to_the_template():
    llm = ScriptedLLM(["He should be declined.", GOOD])
    out = write_explanation(llm, _payload())
    # one call only: no retry, even though a good second reply was waiting
    assert len(llm.prompts) == 1 and len(out["attempts"]) == 1
    assert out["source"] == "template_fallback" and out["fell_back"] and out["text"] == template(_payload())
    assert "failed an automated check" in out["label"]


def test_a_backend_error_falls_back_too():
    class Broken:
        def generate(self, *args, **kwargs):
            raise RuntimeError("Can't reach Ollama")

    out = write_explanation(Broken(), _payload())
    assert out["source"] == "template_fallback" and "Ollama" in out["attempts"][0]["error"]


def test_customers_get_the_template_unless_the_setting_is_on():
    ex = {"decision": "decline", "pd": 0.293, "risk_band": {"band": "E", "label": "very high"},
          "reasons_officer": [{"text": t} for t in _payload()["reasons"]],
          "reasons_customer": [{"text": "Credit bureau score: lowest 10% of applicants"}]}
    customer_text = "Your credit bureau score is in the lowest 10% of applicants."
    out = explain_both(ScriptedLLM([GOOD, customer_text]), ex)
    assert out["officer"]["source"] == "llm" and out["customer"]["source"] == "template"
    out = explain_both(ScriptedLLM([GOOD, customer_text]), ex, customer_llm=True)
    assert out["customer"]["source"] == "llm"
    assert out["customer"]["text"] == "Your application has been declined. " + customer_text


def test_customer_prompt_never_sees_officer_only_reasons_or_the_pd():
    ex = {"decision": "review", "pd": 0.17, "risk_band": {"band": "D", "label": "high"},
          "reasons_officer": [{"text": "Age: 64.9 years (highest 10% of applicants)"}],
          "reasons_customer": [{"text": "Instalments paid on previous loans: 0 (lowest 10% of applicants)"}]}
    from scorecard.narrative import payload_from_explanation

    _, user = build_prompt(payload_from_explanation(ex, "customer"))
    assert "Age" not in user and "17.0%" not in user and "Instalments paid" in user
    _, user = build_prompt(payload_from_explanation(ex, "officer"))
    assert "Age: 64.9" in user and "17.0%" in user


def _client(base_url, handler, headers=None):
    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler), headers=headers or {})


def test_ollama_request_and_parse():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "Text"}, "prompt_eval_count": 50, "eval_count": 5})

    llm = OllamaLLM("http://x", "qwen2.5:3b")
    llm.client = _client("http://x", handler)
    out = llm.generate("sys", "user")
    assert out.text == "Text" and out.input_tokens == 50
    assert seen["messages"][0] == {"role": "system", "content": "sys"} and seen["options"]["temperature"] == 0.0


def test_openai_and_anthropic_request_and_parse():
    def openai_handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}], "usage": {"completion_tokens": 1}})

    def anthropic_handler(request):
        body = json.loads(request.content)
        # this api takes the system prompt in its own field
        assert body["system"] == "s" and body["messages"] == [{"role": "user", "content": "u"}]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Done"}], "usage": {"input_tokens": 3}})

    o = OpenAILLM("key", "gpt-4o-mini", "http://x/v1")
    o.client = _client("http://x/v1", openai_handler)
    assert o.generate("s", "u").output_tokens == 1
    a = AnthropicLLM("key", "claude-haiku-5-5")
    a.client = _client("https://api.anthropic.com", anthropic_handler)
    assert a.generate("s", "u").text == "Done"


def test_backend_choice():
    s = get_settings()
    assert get_llm(s, "none") is None
    with pytest.raises(ValueError):
        get_llm(s.with_overrides(openai_api_key=""), "openai")
    with pytest.raises(ValueError):
        get_llm(s, "gemini")
