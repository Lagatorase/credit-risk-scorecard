# Features

`scripts/build_features.py` builds one row per labelled applicant (307,511 rows, 155 columns, of which 153 are model inputs) from the seven tables. Each side table has its own aggregation function in `scorecard/features.py`. Each feature only uses that applicant's own records and none use the default label, so building them once for everyone can't leak labels between CV folds. Anything learned from labels (WoE bins, calibration) is fit inside each fold.

The "AUC" column is each feature's univariate AUC on all labelled applicants, for orientation only: below 0.5 means higher values go with lower risk.

## Application (83 columns, 81 model inputs, no prefix)

| Group | Examples | Why it should predict default |
|---|---|---|
| External scores | `EXT_SOURCE_1..3`, `EXT_SOURCE_MEAN` (AUC 0.28), `EXT_SOURCE_MIN` | Scores from outside credit bureaus already summarise repayment history elsewhere. They are the strongest single predictors |
| Affordability ratios | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `ANNUITY_CREDIT_RATIO`, `CREDIT_GOODS_RATIO` (0.57), `INCOME_PER_PERSON` | A larger repayment relative to income leaves less room for shocks. Borrowing more than the goods cost (credit/goods above 1) often means fees or insurance were added to the loan |
| Age and employment | `AGE_YEARS` (0.42), `EMPLOYED_YEARS` (0.42), `EMPLOYED_TO_AGE_RATIO`, `DAYS_EMPLOYED_PLACEHOLDER` | Older applicants with longer, stabler employment default less. The placeholder flag marks the 18% with `DAYS_EMPLOYED = 365243`, mostly pensioners |
| Loan and applicant details | `AMT_CREDIT`, `AMT_ANNUITY`, `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` | Loan size, product type, income source and job type all relate to income stability. `CODE_GENDER` and `NAME_FAMILY_STATUS` are in the table but not model inputs (see below) |
| Region and contact | `REGION_RATING_CLIENT`, `REGION_POPULATION_RELATIVE`, `REG_CITY_NOT_WORK_CITY`, `FLAG_EMP_PHONE`, `DAYS_LAST_PHONE_CHANGE` | Regional economic conditions, and signs of instability such as a recent phone change or living away from the work city |
| Housing | 14 `*_AVG` building statistics, `TOTALAREA_MODE`, `FLAG_OWN_REALTY`, `NAME_HOUSING_TYPE` | Housing quality and ownership are rough proxies for wealth. About 50 to 70% missing, and whether they're missing is itself a signal |
| Documents and enquiries | `FLAG_DOCUMENT_3`, `DOCUMENT_COUNT`, `AMT_REQ_CREDIT_BUREAU_*` | Paperwork provided, and how often other lenders recently checked the applicant's credit |
| Social circle | `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE` | Defaults among people in the applicant's social circle |

**Protected attributes.** `CODE_GENDER` and `NAME_FAMILY_STATUS` stay in the feature table but are not inputs to any model in the experiments or the final model (`PROTECTED_INPUTS` in `scorecard/features.py`). Using gender or marital status in a lending decision is hard to justify under MAS's FEAT principles (Fairness, Ethics, Accountability, Transparency). They are kept only for the fairness analysis, which also trains a comparison model with both added back to show what they would add. Age (`AGE_YEARS`, and `EMPLOYED_TO_AGE_RATIO`, which is derived from it) is a model input and can appear in the loan officer's reason codes, but is never given to a customer as a reason.

Cleaning: `DAYS_EMPLOYED = 365243` becomes missing (with a flag), `CODE_GENDER = XNA` (4 rows) becomes missing, and day counts become years. I kept only the AVG version of each housing statistic, since the MODE and MEDI versions mostly repeat it. Of the 20 document flags I kept their count and `FLAG_DOCUMENT_3`, the only common one.

## Credit bureau (24 features, `BUREAU_` and `BB_`)

Loans at other lenders reported to the credit bureau (`bureau`), with the monthly status of each loan (`bureau_balance`, first aggregated to one row per loan).

| Feature | Why it should predict default |
|---|---|
| `BUREAU_LOAN_COUNT`, `BUREAU_ACTIVE_COUNT`, `BUREAU_ACTIVE_SHARE` (0.58) | Many loans still open means more existing obligations |
| `BUREAU_RECENT_2Y_COUNT`, `BUREAU_DAYS_CREDIT_MAX/MIN/MEAN` (mean 0.60) | Recently opened credit, or a short credit history, is riskier than an old, settled one |
| `BUREAU_DEBT_TOTAL`, `BUREAU_ACTIVE_DEBT_RATIO` (0.58), `BUREAU_DEBT_INCOME_RATIO` (0.56), `BUREAU_ACTIVE_ANNUITY_TOTAL` | How much is still owed, against the credit granted and against income |
| `BUREAU_OVERDUE_SHARE`, `BUREAU_DAYS_OVERDUE_MAX`, `BUREAU_OVERDUE_TOTAL`, `BUREAU_MAX_OVERDUE_AMT` | Loans currently or previously in arrears at other lenders |
| `BUREAU_BAD_DEBT_OR_SOLD_COUNT`, `BUREAU_PROLONG_COUNT` | Loans written off or sold, and loans that needed extending |
| `BB_DPD_SHARE_MEAN/MAX`, `BB_WORST_STATUS_MAX`, `BB_RECENT_DPD_SHARE_MAX` | Share of months past due and the worst past-due bucket reached, overall and in the last 12 months |

## Previous Home Credit applications (16 features, `PREV_`)

| Feature | Why it should predict default |
|---|---|
| `PREV_REFUSED_COUNT`, `PREV_REFUSED_SHARE` (0.56), `PREV_REFUSED_RECENT_2Y_COUNT` | Earlier refusals mean underwriters saw risk before |
| `PREV_APP_COUNT`, `PREV_APPROVED_COUNT`, `PREV_RECENT_1Y_COUNT`, `PREV_CANCELED_SHARE` | Frequent recent applications can signal credit hunger |
| `PREV_APP_CREDIT_RATIO_MEAN` (0.45) | Asked-for amount over granted amount: getting what was asked for goes with lower risk |
| `PREV_HIGH_YIELD_SHARE`, `PREV_DOWN_PAYMENT_RATE_MEAN` | High-yield pricing is used for riskier customers, while larger down payments show savings |
| `PREV_AMT_CREDIT_MEAN/MAX`, `PREV_ANNUITY_MEAN`, `PREV_CNT_PAYMENT_MEAN`, `PREV_DAYS_DECISION_MAX/MEAN` | Size, term and timing of earlier loans |

## Instalment payments (10 features, `INST_`)

Partial payments are combined into one row per instalment first, so "late" means the instalment was fully paid after its due date.

| Feature | Why it should predict default |
|---|---|
| `INST_LATE_SHARE` (0.56), `INST_LATE_30_SHARE`, `INST_DAYS_LATE_MEAN/MAX` | Past late payment is the classic predictor of future late payment |
| `INST_RECENT_1Y_LATE_SHARE` (0.57), `INST_RECENT_1Y_UNDERPAID_SHARE` | Recent behaviour matters more than behaviour from years ago |
| `INST_UNDERPAID_SHARE`, `INST_SHORTFALL_TOTAL`, `INST_PAY_RATIO_MEAN` | Paying less than the amount due |
| `INST_COUNT` | Length of repayment history with Home Credit |

## POS and cash loan balances (9 features, `POS_`)

| Feature | Why it should predict default |
|---|---|
| `POS_DPD_SHARE`, `POS_DPD_DEF_SHARE`, `POS_DPD_MAX/MEAN`, `POS_RECENT_1Y_DPD_SHARE` | Months spent past due on earlier point-of-sale and cash loans. `DPD_DEF` ignores tiny overdue amounts |
| `POS_LOAN_COUNT`, `POS_MONTH_COUNT`, `POS_COMPLETED_SHARE` | A longer history of completed loans goes with lower risk |
| `POS_FUTURE_INSTALMENTS_TOTAL` | Instalments still to pay on open loans, so remaining burden |

## Credit card balances (13 features, `CC_`)

Only 28% of applicants had a Home Credit card, so these are mostly missing (LightGBM handles that natively, and WoE gives missing its own bin).

| Feature | Why it should predict default |
|---|---|
| `CC_UTIL_MEAN` (0.63), `CC_UTIL_MAX`, `CC_RECENT_1Y_UTIL_MEAN` (0.64) | Running a card close to its limit is one of the strongest behavioural risk signals |
| `CC_BALANCE_MEAN`, `CC_DRAWINGS_MEAN`, `CC_LIMIT_MAX` | Size of revolving debt and spending |
| `CC_ATM_DRAWINGS_SHARE` | Cash withdrawals on a credit card can signal cash-flow stress |
| `CC_MISSED_MIN_SHARE`, `CC_RECENT_1Y_MISSED_MIN_SHARE`, `CC_DPD_SHARE/MAX` | Months where less than the minimum payment was made, or the card was past due |
| `CC_CARD_COUNT`, `CC_MONTH_COUNT` | Length of card history |

## Notes

- Counts (`*_COUNT`) are 0 for applicants with no records in a table. Every other aggregate is missing for them.
- Money amounts are stored as float32 (about 7 significant digits), which can drop the cents on large loans. That doesn't matter for modelling.
- Some aggregates turned out weak on their own (for example `INST_UNDERPAID_SHARE` at 0.505 and `CC_DPD_SHARE` at 0.496). I kept them, because the models and IV selection decide what is used.
