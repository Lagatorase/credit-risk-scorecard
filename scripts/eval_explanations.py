"""Evaluates the LLM explanations on 50 hold-out applicants with the configured backend (local qwen2.5:3b by default).

  python scripts/eval_explanations.py                 (the final run: new sample, frozen prompts and checks)
  python scripts/eval_explanations.py --report-only   (rebuild the report from the saved run)

The 50 applicants are drawn at random within each decision: 17 declines, 17 reviews and 16 approvals,
so the rarer decisions get enough examples (a plain random draw would be about 85% approvals). The
applicants used while I tuned the prompts and checks (reports/llm_eval_dev.json) are left out, and the
seed differs from the tuning runs, so this is a fresh test.

Both readers get LLM drafts here, including the customer, whose LLM text is off by default in the app
(CUSTOMER_EXPLANATION=template); the eval measures how good that option would be. Reports the pass
rate of every automated check, the fallback rate and failure examples. The development (tuning) runs
are summarised from reports/llm_eval_dev_notes.md. Writes reports/llm_eval.md and reports/llm_eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scorecard.config import get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402
from scorecard.explain import Explainer  # noqa: E402
from scorecard.llm import get_llm  # noqa: E402
from scorecard.narrative import build_prompt, explain_both, payload_from_explanation, template  # noqa: E402

# the tuning runs used seed 42; the final run uses a different one and excludes the tuning applicants
FINAL_SEED = 2026
PER_DECISION = {"decline": 17, "review": 17, "approve": 16}
CHECKS = ["decision", "numbers", "factors", "protected", "officer_only", "coverage", "judgement", "format"]
AUDIENCES = ["officer", "customer"]


def pick_applicants(holdout: pd.DataFrame, exclude: set[int], seed: int) -> pd.DataFrame:
    # stratified by the decision the model makes, so every decision type is tested
    holdout = holdout[~holdout["SK_ID_CURR"].isin(exclude)]
    holdout = holdout.assign(decision=decide(holdout["pd"].to_numpy(), load_decision_config()))
    parts = [holdout[holdout["decision"] == d].sample(n, random_state=seed) for d, n in PER_DECISION.items()]
    return pd.concat(parts)


def rate(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarise(results: list[dict]) -> dict:
    """Pass rate per check and per reader, and the fallback rate (one draft each, no retry)."""
    out = {}
    for aud in AUDIENCES:
        runs = [r["explanations"][aud] for r in results]
        drafts = [run["attempts"][0] for run in runs if run["attempts"] and run["attempts"][0]["checks"]]
        checks = [c for c in CHECKS if c in (drafts[0]["checks"] if drafts else {})]
        out[aud] = {
            "n": len(runs),
            "pass": {c: rate(a["checks"][c]["passed"] for a in drafts) for c in checks},
            "all_pass": rate(all(v["passed"] for v in a["checks"].values()) for a in drafts),
            "fallback_rate": rate(run["fell_back"] for run in runs),
            "errors": sum(1 for run in runs for a in run["attempts"] if a.get("error")),
            "latency_median_s": float(np.median([a["latency_s"] for a in drafts])) if drafts else float("nan"),
        }
    return out


def failure_examples(results: list[dict], limit: int = 8) -> tuple[list[dict], int]:
    rows = []
    for r in results:
        for aud in AUDIENCES:
            for a in r["explanations"][aud]["attempts"]:
                failed = {k: v["detail"] for k, v in a["checks"].items() if not v["passed"]}
                if failed or a.get("error"):
                    rows.append({"id": r["id"], "decision": r["decision"], "audience": aud,
                                 "failed": failed or {"error": a.get("error")}, "text": a["text"]})
    # one per combination of failed checks first, so the examples show the range of problems
    seen, picked = set(), []
    for row in rows:
        key = tuple(sorted(row["failed"]))
        if key not in seen:
            seen.add(key)
            picked.append(row)
    picked += [r for r in rows if r not in picked]
    return picked[:limit], len(rows)


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(reports: Path, data: dict) -> None:
    results, summary = data["results"], data["summary"]
    failures, n_failed = failure_examples(results)
    o, cu = summary["officer"], summary["customer"]
    md = [
        "# LLM explanation eval",
        "",
        f"**Final run.** Backend `{data['backend']}`, temperature 0. {len(results)} hold-out applicants drawn at random "
        f"(seed {data['seed']}) within each decision: {PER_DECISION['decline']} declines, {PER_DECISION['review']} reviews "
        f"and {PER_DECISION['approve']} approvals, none of them used while tuning. The prompts and checks were frozen "
        "before this run, and nothing was changed after it. Each applicant gets an officer and a customer draft, "
        f"{2 * len(results)} drafts in all, in {data['total_seconds'] / 60:.1f} minutes.",
        "",
        "In the app and API the officer sees the LLM draft, labelled \"AI-drafted summary: check against the reason "
        "codes\". The customer gets the plain template by default (`CUSTOMER_EXPLANATION=template`); the customer "
        "drafts here measure the LLM option.",
        "",
        "Each draft is checked automatically (`scorecard/narrative.py`). A draft that fails any check is replaced by a "
        "template built from the reason codes. There is no retry.",
        "",
        "| Check | What it tests |",
        "|---|---|",
        "| decision | The text doesn't contradict the decision (the opening sentence stating it is added by the code) |",
        "| numbers | Every number is one from the reason codes (or the PD, officer only), and each sentence's numbers come "
        "from a single reason |",
        "| factors | No risk factor term (income, employment, debt, ...) that the reason codes given don't mention |",
        "| protected | No gender, marital status, race, religion or nationality terms, and no he/she/his/her |",
        "| officer_only | Customer only: nothing about age, location, contact details or documents |",
        "| coverage | Every reason's value appears in the text (numbers, or the category for categorical reasons) |",
        "| judgement | No judgement words of the model's own (significant, poor, below average, ...) |",
        "| format | Plain prose: no list markers, no prompt text echoed back |",
        "",
        "## Results (final run)",
        "",
        "| Check | Officer | Customer |",
        "|---|---|---|",
    ]
    for c in CHECKS:
        md.append(f"| {c} | " + " | ".join(f"{summary[a]['pass'][c]:.0%}" if c in summary[a]["pass"] else "n/a"
                                           for a in AUDIENCES) + " |")
    md += [
        f"| **all checks pass** | **{o['all_pass']:.0%}** | **{cu['all_pass']:.0%}** |",
        f"| Fell back to the template | {o['fallback_rate']:.0%} | {cu['fallback_rate']:.0%} |",
        f"| Backend errors | {o['errors']} | {cu['errors']} |",
        f"| Median seconds per draft | {o['latency_median_s']:.1f} | {cu['latency_median_s']:.1f} |",
        "",
    ]
    notes = reports / "llm_eval_notes.md"
    if notes.exists():
        md += [notes.read_text(encoding="utf-8").strip(), ""]
    md += [f"## Failure examples ({n_failed} failed drafts)", "",
           "| Applicant | Reader | Failed checks | Draft |", "|---|---|---|---|"]
    for f in failures:
        detail = "; ".join(f"{k}: {v}" for k, v in f["failed"].items())
        md.append(f"| {f['id']} ({f['decision']}) | {f['audience']} | {md_cell(detail)} | {md_cell(f['text'])} |")

    md += ["", "## Example explanations", "",
           "The officer column is what the app shows. The customer column is the LLM option; by default the customer "
           "gets the template shown under it.", ""]
    for decision in ("decline", "review"):
        r = next(x for x in results if x["decision"] == decision)
        e = r["explanations"]
        default_customer = template({"audience": "customer", "decision": decision, "pd": r["pd"],
                                     "band": r["band"]["band"], "band_label": r["band"]["label"],
                                     "reasons": r["reasons_customer"]})
        md += [f"**Applicant {r['id']}**: PD {r['pd']:.1%}, band {r['band']['band']}, decision **{decision}** "
               f"(officer text from: {e['officer']['source']}, customer text from: {e['customer']['source']}).", "",
               "| Loan officer | Customer (LLM option) |", "|---|---|",
               f"| {md_cell(e['officer']['text'])} | {md_cell(e['customer']['text'])} |", "",
               f"Customer default (template): {default_customer}", "",
               "Reason codes given: officer: " + "; ".join(r["reasons_officer"]) + ". Customer: "
               + "; ".join(r["reasons_customer"]) + ".", ""]
    dev = reports / "llm_eval_dev_notes.md"
    if dev.exists():
        md += ["## Development results (tuning runs, not the headline)", "",
               "These come from the runs I used to design the prompts and checks, on a different 50 applicants "
               "(seed 42), with a feedback retry that has since been removed. Full detail of the last tuning run is in "
               "`reports/llm_eval_dev.md`.", "", dev.read_text(encoding="utf-8").strip().replace("## ", "### "), ""]
    (reports / "llm_eval.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    if "--report-only" in sys.argv:
        # rebuilds the markdown from the saved run, e.g. after writing the notes
        write_report(reports, json.loads((reports / "llm_eval.json").read_text()))
        return
    llm = get_llm(settings)
    if llm is None:
        raise SystemExit("LLM_BACKEND is none; set it to ollama, openai or anthropic to evaluate the explanations")

    dev_ids = {r["id"] for r in json.loads((reports / "llm_eval_dev.json").read_text())["results"]}
    holdout = pd.read_parquet(reports / "cache" / "holdout_pd.parquet")
    chosen = pick_applicants(holdout, dev_ids, FINAL_SEED)
    ids = chosen["SK_ID_CURR"].tolist()
    assert not set(ids) & dev_ids
    features = pd.read_parquet(settings.processed_dir / "features.parquet")
    rows = features.set_index("SK_ID_CURR").loc[ids].reset_index()
    del features

    explainer = Explainer.from_files(settings.models_dir)
    explanations = explainer.explain(rows)

    # one untimed call first, so the model loading into GPU memory doesn't count against the first applicant
    system, user = build_prompt(payload_from_explanation(explanations[0], "officer"))
    llm.generate(system, user)

    results, start = [], time.perf_counter()
    for app_id, target, ex in zip(ids, rows["TARGET"], explanations):
        out = explain_both(llm, ex, customer_llm=True)
        results.append({"id": int(app_id), "target": int(target), "decision": ex["decision"], "pd": ex["pd"],
                        "band": ex["risk_band"], "reasons_officer": [r["text"] for r in ex["reasons_officer"]],
                        "reasons_customer": [r["text"] for r in ex["reasons_customer"]], "explanations": out})
        print(f"{len(results):>2}/50 {app_id} {ex['decision']:<8} officer: {out['officer']['source']:<17} "
              f"customer: {out['customer']['source']}")
    data = {"backend": llm.name, "seed": FINAL_SEED, "summary": summarise(results),
            "total_seconds": time.perf_counter() - start, "results": results}
    (reports / "llm_eval.json").write_text(json.dumps(data, indent=2, default=float))
    write_report(reports, data)


if __name__ == "__main__":
    main()
