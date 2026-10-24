"""Streamlit demo: score a hold-out applicant, see why, and try what-if changes.

  streamlit run app/streamlit_app.py

Needs models/final_model.joblib, models/reference.joblib and models/demo_applicants.parquet, built by
scripts/train_final.py, scripts/build_reference.py and scripts/export_demo_applicants.py. The app scores
in-process with the same ScoringService as the API, so the what-if sliders update without a round trip.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from scorecard.config import get_settings  # noqa: E402
from scorecard.reasons import format_value  # noqa: E402
from scorecard.service import EDITABLE, ScoringService, apply_whatif, model_card_text  # noqa: E402

st.set_page_config(page_title="Credit Risk Scorecard", page_icon=":material/credit_score:", layout="wide")

# one colour per decision, used for the badge and the gauge; white text on these fills reads in both themes
DECISIONS = {
    "approve": ("Approve", "#2E7D4F"),
    "review": ("Refer to a loan officer", "#B7791F"),
    "decline": ("Decline", "#B83A3A"),
}
NAVY = "#1F3A5F"
TEAL = "#1C7C8C"
# the inputs shown in the applicant summary, with how to format them
SUMMARY = [
    ("AMT_CREDIT", "Loan amount", "money"),
    ("AMT_GOODS_PRICE", "Price of the goods", "money"),
    ("AMT_ANNUITY", "Yearly repayment", "money"),
    ("AMT_INCOME_TOTAL", "Yearly income", "money"),
    ("EMPLOYED_YEARS", "Years in current job", "years"),
    ("EXT_SOURCE_MEAN", "Average bureau score", "score"),
]

CSS = """
<style>
#MainMenu, footer, .stAppDeployButton, [data-testid="stStatusWidget"], header[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 10px;}
.app-header {background: #1F3A5F; color: #FFFFFF; border-radius: 10px; padding: 16px 22px; margin-bottom: 18px;
  display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;}
.app-header h1 {font-size: 1.45rem; margin: 0; padding: 0; color: #FFFFFF; font-weight: 650;}
.app-header p {margin: 2px 0 0 0; opacity: 0.85; font-size: 0.92rem;}
.chips {display: flex; gap: 8px; flex-wrap: wrap;}
.chip {background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
  padding: 3px 10px; font-size: 0.78rem; white-space: nowrap;}
.section-title {font-size: 0.78rem; font-weight: 650; letter-spacing: 0.04em; text-transform: uppercase;
  opacity: 0.65; margin: 0 0 8px 0;}
.kv {display: grid; grid-template-columns: 1fr auto; row-gap: 6px; column-gap: 12px; font-size: 0.93rem;}
.kv .k {opacity: 0.72;} .kv .v {font-variant-numeric: tabular-nums; text-align: right; font-weight: 550;}
.kv .changed {color: #1C7C8C; font-weight: 650;}
.pd-row {display: flex; align-items: center; gap: 14px; flex-wrap: wrap;}
.pd-value {font-size: 3.1rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums;}
.pd-caption {font-size: 0.85rem; opacity: 0.7; margin-top: 4px;}
.badge {display: inline-block; color: #FFFFFF; border-radius: 6px; padding: 5px 12px; font-weight: 650;
  font-size: 0.95rem;}
.band {display: inline-block; white-space: nowrap; border: 1px solid rgba(128,128,128,0.45); border-radius: 6px; padding: 4px 10px;
  font-size: 0.9rem;}
.delta {margin: 10px 0 18px 0; font-size: 0.95rem; font-weight: 550;}
.reason {display: grid; grid-template-columns: 22px 1fr 120px; gap: 10px; align-items: center; padding: 8px 0;
  border-top: 1px solid rgba(128,128,128,0.2); font-size: 0.92rem;}
.reason:first-of-type {border-top: none;}
.reason .rank {opacity: 0.55; font-weight: 650;}
.reason .bar {height: 8px; border-radius: 4px; background: #B83A3A; opacity: 0.85;}
.reason .impact {font-size: 0.75rem; opacity: 0.65; margin-top: 3px; font-variant-numeric: tabular-nums;}
.text-label {display: inline-block; font-size: 0.78rem; font-weight: 600; border-radius: 6px; padding: 3px 9px;
  margin-bottom: 8px; border: 1px solid rgba(128,128,128,0.45);}
.text-label.ai {border-color: #B7791F; color: #B7791F;}
.explanation {font-size: 0.97rem; line-height: 1.55;}
.muted {opacity: 0.7; font-size: 0.85rem;}
/* below laptop width the two columns stack, so the charts and reason text keep their room */
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] {flex-direction: column; gap: 1rem;}
  [data-testid="stColumn"] {width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;}
}
</style>
"""


@st.cache_resource(show_spinner="Loading the model")
def load_service() -> ScoringService:
    return ScoringService.from_settings(get_settings())


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return pd.read_parquet(get_settings().models_dir / "demo_applicants.parquet").set_index("SK_ID_CURR")


@st.cache_data(show_spinner=False)
def original_scores() -> dict:
    # every demo applicant scored once without text, so the picker can show their decision
    service, demo = load_service(), load_demo()
    return {int(i): service.score(row, explanations=False, include_shap=True) for i, row in demo.iterrows()}


@st.cache_data(show_spinner="Writing the explanations")
def original_explanations(app_id: int) -> dict:
    # the LLM runs once per applicant and the result is cached, so switching back is instant
    return load_service().score(load_demo().loc[app_id])["explanations"]


def money(v) -> str:
    return "Not recorded" if v is None or pd.isna(v) else f"{v:,.0f}"


def years(v) -> str:
    return "Not recorded" if v is None or pd.isna(v) else f"{v:.1f} years"


def slider_options(feature: str, original) -> list:
    """Values the what-if slider can take: a round-number grid that also contains the original value."""
    if feature == "EMPLOYED_YEARS":
        grid = list(np.round(np.arange(0, 40.5, 0.5), 1))
    else:
        step, top = {"AMT_CREDIT": (10_000, 2_500_000), "AMT_GOODS_PRICE": (10_000, 2_500_000),
                     "AMT_ANNUITY": (1_000, 120_000), "AMT_INCOME_TOTAL": (5_000, 800_000)}[feature]
        top = max(top, (original or 0) * 1.5)
        grid = list(np.arange(step, top + step, step, dtype=float))
    if original is None or pd.isna(original):
        # an applicant without a value (no current job) starts at "Not recorded", shown as None
        return [None, *grid]
    value = round(float(original), 1) if feature == "EMPLOYED_YEARS" else float(original)
    return sorted(set(grid) | {value})


def reset_whatif(app_id: int) -> None:
    for f in EDITABLE:
        st.session_state.pop(f"wi_{app_id}_{f}", None)


def gauge(pd_value: float, decision: str, review: tuple[float, float]) -> go.Figure:
    pct = pd_value * 100
    top = max(40.0, np.ceil(pct / 10) * 10)
    lo, hi = review[0] * 100, review[1] * 100
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=pct,
        gauge={
            "shape": "angular",
            "axis": {"range": [0, top], "ticksuffix": "%", "tickwidth": 1, "nticks": 6},
            "bar": {"color": DECISIONS[decision][1], "thickness": 0.32},
            "borderwidth": 0,
            "steps": [
                {"range": [0, lo], "color": "rgba(46,125,79,0.20)"},
                {"range": [lo, hi], "color": "rgba(183,121,31,0.26)"},
                {"range": [hi, top], "color": "rgba(184,58,58,0.20)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin={"l": 34, "r": 34, "t": 20, "b": 18},
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def wrap(text: str, width: int = 42) -> str:
    # long axis labels break onto a second line instead of squeezing the bars
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "<br>".join([*lines, line])


def waterfall(result: dict, row: pd.Series, coder, n: int = 10) -> go.Figure:
    """The applicant's largest SHAP contributions, from the average applicant's log-odds to their own."""
    shap = pd.Series(result["shap"])
    top = shap.abs().sort_values(ascending=False).index[:n]
    labels = []
    for f in top:
        entry = coder.features[f]
        value = row.get(f)
        shown = "not available" if value is None or pd.isna(value) else format_value(
            value, entry["format"], coder.currency, entry.get("unit"))
        labels.append(wrap(f"{entry['officer']} = {shown}"))
    base, total = result["base_value"], result["base_value"] + shap.sum()
    names = [*labels, f"Other {len(shap) - n} inputs", "This applicant"]
    steps = [*shap[top].to_numpy(), shap.drop(top).sum(), 0.0]
    measure = ["relative"] * (n + 1) + ["total"]
    # the bars start at the average applicant's log-odds and the total bar ends at this applicant's
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=steps, base=base, measure=measure,
        text=[f"{v:+.2f}" for v in steps[:-1]] + [f"{total:.2f}"], textposition="outside", cliponaxis=False,
        increasing={"marker": {"color": "#B83A3A"}},
        decreasing={"marker": {"color": "#2E7D4F"}},
        totals={"marker": {"color": "#5B7393"}},
        connector={"line": {"color": "rgba(128,128,128,0.4)", "width": 1}},
    ))
    fig.add_vline(x=base, line={"color": "rgba(128,128,128,0.7)", "dash": "dot", "width": 1})
    fig.add_annotation(x=base, y=1.0, yref="paper", yanchor="bottom", showarrow=False, font={"size": 11},
                       text=f"average applicant {base:.2f}")
    fig.update_layout(height=38 * (n + 2) + 90, margin={"l": 10, "r": 50, "t": 30, "b": 40},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Log-odds of default (raw model, before calibration)", showlegend=False)
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def reasons_html(reasons: list[dict]) -> str:
    if not reasons:
        return "<p class='muted'>No input pushed this applicant's risk up.</p>"
    biggest = max(r["impact"] for r in reasons)
    rows = []
    for i, r in enumerate(reasons, 1):
        width = max(6, 100 * r["impact"] / biggest)
        rows.append(f"<div class='reason'><div class='rank'>{i}</div><div>{escape(r['text'])}</div>"
                    f"<div><div class='bar' style='width:{width:.0f}%'></div>"
                    f"<div class='impact'>+{r['impact']:.2f} log-odds</div></div></div>")
    return "".join(rows)


def explanation_block(text: dict, ai: bool) -> None:
    css = "text-label ai" if ai and text["source"] == "llm" else "text-label"
    st.markdown(f"<span class='{css}'>{escape(text['label'])}</span>"
                f"<div class='explanation'>{escape(text['text'])}</div>", unsafe_allow_html=True)


# ---- page ----

st.markdown(CSS, unsafe_allow_html=True)
settings = get_settings()
try:
    service = load_service()
    demo = load_demo()
except FileNotFoundError as exc:
    st.error(f"The model or the demo applicants are missing ({exc}). See the README for how to build them.")
    st.stop()
scores = original_scores()
cfg = service.explainer.cfg
llm_name = service.llm.name if service.llm else "none (template only)"

st.markdown(
    "<div class='app-header'><div><h1>Credit Risk Scorecard</h1>"
    "<p>Probability of default, an approve, review or decline decision, and the reasons behind it.</p></div>"
    "<div class='chips'><span class='chip'>LightGBM, 153 inputs, Platt calibrated</span>"
    "<span class='chip'>Hold-out AUC 0.791</span>"
    f"<span class='chip'>LLM: {escape(llm_name)}</span></div></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 7], gap="large")

with left:
    with st.container(border=True):
        st.markdown("<div class='section-title'>Applicant</div>", unsafe_allow_html=True)
        ids = list(scores)
        app_id = st.selectbox(
            "Hold-out applicant", ids, label_visibility="collapsed",
            format_func=lambda i: f"Applicant {i}  ·  {scores[i]['decision']}",
        )
        original_row = demo.loc[app_id]

        st.markdown("<div class='section-title' style='margin-top:14px'>What if</div>", unsafe_allow_html=True)
        st.caption("Change the loan or the applicant's income and job. Ratios built from these inputs are "
                   "recomputed, everything else stays as recorded.")
        changes = {}
        for feature, label in EDITABLE.items():
            original = original_row.get(feature)
            options = slider_options(feature, original)
            start = None if original is None or pd.isna(original) else (
                round(float(original), 1) if feature == "EMPLOYED_YEARS" else float(original))
            fmt = years if feature == "EMPLOYED_YEARS" else money
            chosen = st.select_slider(label, options=options, value=start, format_func=fmt,
                                      key=f"wi_{app_id}_{feature}")
            if chosen != start and chosen is not None:
                changes[feature] = chosen
        st.button("Reset to the recorded values", on_click=reset_whatif, args=(app_id,),
                  disabled=not changes, width="stretch")

    row = apply_whatif(original_row, changes) if changes else original_row
    with st.container(border=True):
        st.markdown("<div class='section-title'>Applicant summary</div>", unsafe_allow_html=True)
        cells = []
        for feature, label, kind in SUMMARY:
            value = row.get(feature)
            shown = (money(value) if kind == "money" else years(value) if kind == "years"
                     else "Not recorded" if pd.isna(value) else f"{value:.2f}")
            # values changed in the what-if are shown in the accent colour
            css = "v changed" if feature in changes else "v"
            cells.append(f"<div class='k'>{label}</div><div class='{css}'>{shown}</div>")
        st.markdown(f"<div class='kv'>{''.join(cells)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='muted' style='margin-top:10px'>A random hold-out applicant from Home Credit's data, "
                    "never seen in training. Amounts are in Home Credit's unstated currency.</div>",
                    unsafe_allow_html=True)

base = scores[app_id]
result = service.score(row, explanations=False, include_shap=True) if changes else base
decision_text, colour = DECISIONS[result["decision"]]
lo, hi = cfg.review_range

with right:
    with st.container(border=True):
        top_left, top_right = st.columns([5, 6], gap="medium")
        with top_left:
            st.markdown("<div class='section-title'>Probability of default</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='pd-row'><div class='pd-value'>{result['pd']:.1%}</div></div>"
                f"<div class='pd-caption'>chance of payment difficulties on this loan</div>"
                f"<div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap'>"
                f"<span class='badge' style='background:{colour}'>{decision_text}</span>"
                f"<span class='band'>Risk band {result['risk_band']['band']}: {result['risk_band']['label']}</span>"
                f"</div>"
                f"<div class='pd-caption' style='margin-top:12px'>Approve below {lo:.1%}, review from {lo:.1%} to "
                f"{hi:.1%}, decline from {hi:.1%}</div>",
                unsafe_allow_html=True,
            )
            if changes:
                diff = (result["pd"] - base["pd"]) * 100
                direction = "up" if diff > 0 else "down"
                moved = f"{direction} {abs(diff):.1f} points" if abs(diff) >= 0.05 else "no change"
                st.markdown(f"<div class='delta'>PD {result['pd']:.1%}, {moved} from {base['pd']:.1%} "
                            f"for the recorded application ({base['decision']})</div>", unsafe_allow_html=True)
        with top_right:
            st.plotly_chart(gauge(result["pd"], result["decision"], cfg.review_range), width="stretch",
                            config={"displayModeBar": False}, theme="streamlit")

        st.markdown("<div class='section-title' style='margin-top:6px'>Main reasons for the risk</div>",
                    unsafe_allow_html=True)
        audience = st.segmented_control("Reason codes for", ["Loan officer", "Customer"], default="Loan officer",
                                        label_visibility="collapsed") or "Loan officer"
        key = "reasons_officer" if audience == "Loan officer" else "reasons_customer"
        st.markdown(reasons_html(result[key]), unsafe_allow_html=True)
        st.caption("Bar length shows how much each factor raised the risk score (in log-odds; the average "
                   "applicant is the starting point).")
        if audience == "Customer":
            st.caption("The customer's list leaves out officer-only inputs (age, location, contact and document "
                       "details) and shows bureau scores only as a percentile.")

    tab_shap, tab_text, tab_card = st.tabs(["SHAP waterfall", "Explanations", "Model card"])

    with tab_shap:
        st.plotly_chart(waterfall(result, row, service.explainer.coder), width="stretch",
                        config={"displayModeBar": False}, theme="streamlit")
        st.caption(f"The bars start at {result['base_value']:.2f}, the log-odds of the average development "
                   "applicant before calibration, and end at this applicant's own log-odds.")
        st.caption("Each bar is one input's SHAP value: how far it moves this applicant's log-odds from the "
                   "average applicant's. Red raises the risk, green lowers it. The PD above is these log-odds "
                   "after Platt calibration, which keeps the order.")

    with tab_text:
        if changes:
            scenario_key = (app_id, tuple(sorted(changes.items())))
            saved = st.session_state.setdefault("scenario_texts", {})
            if scenario_key not in saved:
                st.info("These inputs differ from the recorded application. The explanations are written "
                        "on request, because the LLM takes a second or two.")
                if st.button("Write explanations for this scenario", type="primary"):
                    with st.spinner("Writing the explanations"):
                        saved[scenario_key] = service.score(row)["explanations"]
            texts = saved.get(scenario_key)
        else:
            texts = original_explanations(app_id)
        if texts:
            officer_col, customer_col = st.columns(2, gap="medium")
            with officer_col:
                st.markdown("<div class='section-title'>Loan officer</div>", unsafe_allow_html=True)
                explanation_block(texts["officer"], ai=True)
            with customer_col:
                st.markdown("<div class='section-title'>Customer</div>", unsafe_allow_html=True)
                explanation_block(texts["customer"], ai=False)
            st.caption("The officer sees a checked AI draft. If it fails any automated check (numbers, factors, "
                       "protected terms, coverage) the template is shown instead. Customers get the template by "
                       "default: in the eval about 1 in 5 accepted customer drafts misread a reason.")

    with tab_card:
        st.markdown(model_card_text())
