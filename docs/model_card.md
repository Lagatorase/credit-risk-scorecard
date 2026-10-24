# Model card: credit risk PD model

## What it does

Estimates the probability that a consumer loan applicant will have payment difficulties (Home Credit's `TARGET`: a late payment of more than a set number of days on at least one of the first instalments). The PD feeds a cost-based rule that approves, sends to manual review, or declines, and every score comes with reason codes for the loan officer and the customer.

This is a portfolio project on public competition data. It has not been validated for real lending.

## Model

- **Algorithm:** LightGBM, 1,160 trees, learning rate 0.046, 27 leaves (the best of five Optuna parameter sets by 5-fold CV AUC).
- **Calibration:** Platt scaling fitted on out-of-fold predictions only.
- **Inputs:** 153 features from the application, credit bureau, previous application, instalment, POS/cash and credit card tables (`docs/features.md`).
- **Excluded on purpose:** `CODE_GENDER` and `NAME_FAMILY_STATUS`. Adding them back gave no measurable accuracy gain and widened the gender approval gap.
- **Kept on purpose:** age. Removing it didn't stop age-based sorting (the other inputs identify under-25s with AUC 0.993), and it lowered under-25s' approval rate from 70.3% to 66.7%.

## Data

Home Credit Default Risk (Kaggle), 307,511 labelled applicants with a default rate of 8.07%. A fixed stratified split (`data/splits.json`) gives 246,008 development applicants and 61,503 in the hold-out set. All model comparison used 5-fold stratified CV on the development set. The hold-out set was scored once, for the final model.

## Performance (hold-out, 95% bootstrap CIs)

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.7908 | 0.7843 to 0.7976 |
| Gini | 0.5815 | 0.5686 to 0.5953 |
| KS | 0.4403 | 0.4294 to 0.4549 |
| Brier | 0.06536 | 0.06396 to 0.06691 |
| Calibration slope (ideal 1) | 1.0077 | 0.9787 to 1.0381 |
| Calibration intercept (ideal 0) | 0.0015 | -0.0286 to 0.0294 |
| PSI of the PD, development vs hold-out | 0.0004 | stable |

The PSI only confirms the split is consistent. The data has no application dates, so there's no out-of-time test.

## Decision rule

The rule assumes an LGD of 45% and a 10% margin lost on a declined good applicant (assumptions, not bank figures), which gives a break-even PD of 18.2%. The rule approves below 14.5%, refers 14.5% to 21.8% to manual review, and declines above.

| Decision | Hold-out share | Actual default rate |
|---|---|---|
| Approve | 84.7% | 4.8% |
| Review | 7.3% | 17.7% |
| Decline | 8.0% | 34.0% |

Risk bands: A below 2.5%, B below 5%, C below 10%, D below 20%, E above.

## Explanations

- **Reason codes:** the inputs with the largest positive SHAP values, at most one per family. Each states the applicant's value and where it sits among development applicants.
- **Officer-only inputs:** age, location, contact, document and social circle inputs never appear in a customer's reasons.
- **Bureau scores for customers:** customers see only the percentile.
- **Written text:** the officer gets an LLM draft (qwen2.5:3b by default), labelled as an AI draft, that must pass automated checks or is replaced by a template. Customers get the template by default.
- **Final eval of the written text (50 new hold-out applicants):** 80% of officer drafts passed all checks. My manual read found a misstatement in 1 of the 40 accepted officer texts and in 5 of the 25 accepted customer texts.

## Fairness

- **Gender:** the main model approves 80.3% of men and 87.0% of women. Men in the hold-out set defaulted more often (10.2% against 7.0%), but the gap remains among applicants who repaid (84.3% against 89.4%). Gender can be predicted from the model's own inputs with AUC 0.904.
- **Age:** approval runs from 70.3% for under-25s to 98.2% for over-65s, in line with default rates of 11.7% and 3.6%. The model is calibrated within every age band.

Full detail is in `reports/fairness_report.md`.

## Limits

- The model learns from one lender's past approvals only: applicants Home Credit refused have no outcome.
- The currency of the amounts is not stated in the data.
- The decision costs are assumptions.
- The LLM text isn't fully reproducible, even at temperature 0.
- The fairness checks cover only the attributes the data records: gender as two values, age and marital status.
