"""Evaluates the LLM explanations on 50 hold-out applicants with the configured backend (local qwen2.5:3b by default).

  python scripts/eval_explanations.py
  python scripts/eval_explanations.py --report-only   (rebuild the report from the saved run)

The 50 applicants are drawn at random (seed 42) within each decision: 17 declines, 17 reviews and 16
approvals, so the rarer decisions get enough examples (a plain random draw would be about 85% approvals).
Each applicant gets an officer and a customer explanation. Reports the pass rate of every automated
check on the first draft and on all drafts, the retry and fallback rates, failure examples and two
example explanations. Writes reports/llm_eval.md and reports/llm_eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scorecard.config import SEED, get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402
from scorecard.explain import Explainer  # noqa: E402
from scorecard.llm import get_llm  # noqa: E402
from scorecard.narrative import build_prompt, explain_both, payload_from_explanation  # noqa: E402

PER_DECISION = {"decline": 17, "review": 17, "approve": 16}
CHECKS = ["decision", "numbers", "factors", "protected", "officer_only", "coverage", "judgement", "format"]
AUDIENCES = ["officer", "customer"]


def pick_applicants(holdout: pd.DataFrame) -> pd.DataFrame:
    # stratified by the decision the model makes, so every decision type is tested
    holdout = holdout.assign(decision=decide(holdout["pd"].to_numpy(), load_decision_config()))
    parts = [holdout[holdout["decision"] == d].sample(n, random_state=SEED) for d, n in PER_DECISION.items()]
    return pd.concat(parts)


def rate(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarise(results: list[dict]) -> dict:
    """Pass rates per check and per reader, plus retry and fallback rates."""
    out = {}
    for aud in AUDIENCES:
        runs = [r["explanations"][aud] for r in results]
        firsts = [run["attempts"][0] for run in runs if run["attempts"] and run["attempts"][0]["checks"]]
        drafts = [a for run in runs for a in run["attempts"] if a["checks"]]
        checks = [c for c in CHECKS if c in (firsts[0]["checks"] if firsts else {})]
        out[aud] = {
            "n": len(runs),
            "first_draft_pass": {c: rate(a["checks"][c]["passed"] for a in firsts) for c in checks},
            "first_draft_all_pass": rate(all(v["passed"] for v in a["checks"].values()) for a in firsts),
            "all_drafts_pass": {c: rate(a["checks"][c]["passed"] for a in drafts) for c in checks},
            "n_drafts": len(drafts),
            "retry_rate": rate(run["retried"] for run in runs),
            "retry_success": rate(run["source"] == "llm_retry" for run in runs if run["retried"]),
            "fallback_rate": rate(run["fell_back"] for run in runs),
            "errors": sum(1 for run in runs for a in run["attempts"] if a.get("error")),
            "sources": pd.Series([run["source"] for run in runs]).value_counts().to_dict(),
            "latency_median_s": float(np.median([a["latency_s"] for a in drafts])) if drafts else float("nan"),
        }
    return out


def failure_examples(results: list[dict], limit: int = 8) -> tuple[list[dict], int]:
    rows = []
    for r in results:
        for aud in AUDIENCES:
            for i, a in enumerate(r["explanations"][aud]["attempts"]):
                failed = {k: v["detail"] for k, v in a["checks"].items() if not v["passed"]}
                if failed or a.get("error"):
                    rows.append({"id": r["id"], "decision": r["decision"], "audience": aud, "attempt": i + 1,
                                 "failed": failed or {"error": a.get("error")}, "text": a["text"]})
    # one per failed check type first, so the examples show the range of problems
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


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    if "--report-only" in sys.argv:
        # rebuilds the markdown from the saved run, e.g. after editing the notes
        write_report(reports, json.loads((reports / "llm_eval.json").read_text()))
        return
    llm = get_llm(settings)
    if llm is None:
        raise SystemExit("LLM_BACKEND is none; set it to ollama, openai or anthropic to evaluate the explanations")

    holdout = pd.read_parquet(reports / "cache" / "holdout_pd.parquet")
    chosen = pick_applicants(holdout)
    ids = chosen["SK_ID_CURR"].tolist()
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
        out = explain_both(llm, ex)
        results.append({"id": int(app_id), "target": int(target), "decision": ex["decision"], "pd": ex["pd"],
                        "band": ex["risk_band"], "reasons_officer": [r["text"] for r in ex["reasons_officer"]],
                        "reasons_customer": [r["text"] for r in ex["reasons_customer"]], "explanations": out})
        print(f"{len(results):>2}/50 {app_id} {ex['decision']:<8} officer: {out['officer']['source']:<17} "
              f"customer: {out['customer']['source']}")
    total_s = time.perf_counter() - start

    data = {"backend": llm.name, "summary": summarise(results), "total_seconds": total_s, "results": results}
    (reports / "llm_eval.json").write_text(json.dumps(data, indent=2, default=float))
    write_report(reports, data)


def write_report(reports: Path, data: dict) -> None:
    results, summary, total_s = data["results"], data["summary"], data["total_seconds"]
    failures, n_failed_drafts = failure_examples(results)
    md = [
        "# LLM explanation eval",
        "",
        f"Backend: `{data['backend']}`, temperature 0. {len(results)} hold-out applicants drawn at random (seed {SEED}) within "
        f"each decision: {PER_DECISION['decline']} declines, {PER_DECISION['review']} reviews and "
        f"{PER_DECISION['approve']} approvals. Each gets an officer and a customer explanation, so "
        f"{2 * len(results)} explanations in all. Total time {total_s / 60:.1f} minutes.",
        "",
        "Each draft is checked automatically (`scorecard/narrative.py`). A draft that fails any check is retried once "
        "with the failures quoted back to the model; if the retry fails as well, the reader gets a plain template built "
        "from the reason codes.",
        "",
        "| Check | What it tests |",
        "|---|---|",
        "| decision | The rest of the text doesn't contradict the decision (the opening sentence stating it is added by the code) |",
        "| numbers | Every number is one from the reason codes (or the PD, officer only), and each sentence's numbers come from a single reason |",
        "| factors | No risk factor term (income, employment, debt, ...) that the reason codes given don't mention |",
        "| protected | No gender, marital status, race, religion or nationality terms, and no he/she/his/her |",
        "| officer_only | Customer only: nothing about age, contact details or documents |",
        "| coverage | Every reason's value appears in the text (numbers, or the category for categorical reasons) |",
        "| judgement | No judgement words of the model's own (significant, poor, below average, ...) |",
        "| format | Plain prose: no list markers, no prompt text or retry feedback echoed back |",
        "",
        "## Pass rates",
        "",
        "| Check | Officer: first draft | Officer: all drafts | Customer: first draft | Customer: all drafts |",
        "|---|---|---|---|---|",
    ]
    for c in CHECKS:
        cells = []
        for aud in AUDIENCES:
            s = summary[aud]
            if c in s["first_draft_pass"]:
                cells += [f"{s['first_draft_pass'][c]:.0%}", f"{s['all_drafts_pass'][c]:.0%}"]
            else:
                cells += ["n/a", "n/a"]
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    o, cu = summary["officer"], summary["customer"]
    md += [
        f"| **all checks** | **{o['first_draft_all_pass']:.0%}** | | **{cu['first_draft_all_pass']:.0%}** | |",
        "",
        "\"All drafts\" counts first drafts and retries together "
        f"({o['n_drafts']} officer drafts, {cu['n_drafts']} customer drafts).",
        "",
        "| | Officer | Customer |",
        "|---|---|---|",
        f"| Retried | {o['retry_rate']:.0%} | {cu['retry_rate']:.0%} |",
        f"| Retry fixed it | {o['retry_success']:.0%} | {cu['retry_success']:.0%} |",
        f"| Fell back to the template | {o['fallback_rate']:.0%} | {cu['fallback_rate']:.0%} |",
        f"| Backend errors | {o['errors']} | {cu['errors']} |",
        f"| Median seconds per draft | {o['latency_median_s']:.1f} | {cu['latency_median_s']:.1f} |",
        "",
        f"## Failure examples ({n_failed_drafts} failed drafts in all)",
        "",
        "| Applicant | Reader | Attempt | Failed checks | Draft |",
        "|---|---|---|---|---|",
    ]
    for f in failures:
        detail = "; ".join(f"{k}: {v}" for k, v in f["failed"].items())
        md.append(f"| {f['id']} ({f['decision']}) | {f['audience']} | {f['attempt']} | {md_cell(detail)} | "
                  f"{md_cell(f['text'])} |")

    md += ["", "## Example explanations", ""]
    for decision in ("decline", "review"):
        r = next(x for x in results if x["decision"] == decision)
        e = r["explanations"]
        md += [f"**Applicant {r['id']}**: PD {r['pd']:.1%}, band {r['band']['band']}, decision **{decision}** "
               f"(officer text from: {e['officer']['source']}, customer text from: {e['customer']['source']}).", "",
               "| Loan officer | Customer |", "|---|---|",
               f"| {md_cell(e['officer']['text'])} | {md_cell(e['customer']['text'])} |", "",
               "Reason codes given: officer: " + "; ".join(r["reasons_officer"]) + ". Customer: "
               + "; ".join(r["reasons_customer"]) + ".", ""]
    notes = reports / "llm_eval_notes.md"
    if notes.exists():
        md += [notes.read_text(encoding="utf-8").strip(), ""]
    (reports / "llm_eval.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
