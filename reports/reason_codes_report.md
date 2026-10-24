# Explainability and reason codes

SHAP values come from `shap.TreeExplainer` on the final LightGBM booster, in log-odds of default before calibration (Platt scaling is monotonic, so the direction of every factor is the same for the calibrated PD).

![SHAP summary](figures/shap_summary.png)

## Top 10 features by mean absolute SHAP value

| Feature | Mean abs SHAP |
|---|---|
| `EXT_SOURCE_MEAN` | 0.321 |
| `ANNUITY_CREDIT_RATIO` | 0.103 |
| `PREV_ANNUITY_MEAN` | 0.102 |
| `EXT_SOURCE_1` | 0.102 |
| `ORGANIZATION_TYPE` | 0.101 |
| `CREDIT_GOODS_RATIO` | 0.099 |
| `EXT_SOURCE_3` | 0.098 |
| `POS_FUTURE_INSTALMENTS_TOTAL` | 0.097 |
| `AMT_ANNUITY` | 0.096 |
| `EXT_SOURCE_2` | 0.090 |

## How reason codes are built

For each applicant I take the features with the largest positive SHAP values (the ones that pushed the PD up), up to four, and turn each into text with `config/reason_codes.yaml`. Numeric features say high or low against the development median, flags say yes or no, and categorical features name the category. The loan officer sees all factors. The customer version skips age-based factors (`AGE_YEARS`, `EMPLOYED_TO_AGE_RATIO`) and takes the next factor instead. Gender and marital status are not model inputs, so they can never appear.

On the 5,000 sampled hold-out applicants, an age-based factor was in the officer's top four for 5.0% of applicants, and customers got 4.00 reasons on average.

Most frequent officer reasons in the sample:

| Feature | Applicants (of 5,000) |
|---|---|
| `EXT_SOURCE_MEAN` | 1,876 |
| `AMT_ANNUITY` | 1,177 |
| `PREV_ANNUITY_MEAN` | 1,103 |
| `CREDIT_GOODS_RATIO` | 815 |
| `ANNUITY_CREDIT_RATIO` | 788 |
| `EXT_SOURCE_1` | 767 |
| `EXT_SOURCE_3` | 767 |
| `ORGANIZATION_TYPE` | 766 |
| `OCCUPATION_TYPE` | 702 |
| `PREV_CNT_PAYMENT_MEAN` | 694 |

## Examples (random hold-out applicants)

**Applicant 323877**: PD 25.1%, band E (very high), decision **decline** (actual outcome: repaid).

| Loan officer reasons | Customer reasons |
|---|---|
| No score from external credit bureau (source 2) (`EXT_SOURCE_2`, +0.26) | No score from external credit bureau (source 2) |
| Lives in a city and region with a weaker risk rating (`REGION_RATING_CLIENT_W_CITY`, +0.23) | Lives in a city and region with a weaker risk rating |
| Low average repayment on previous loans (`PREV_ANNUITY_MEAN`, +0.18) | Low average repayment on previous loans |
| Long repayment term (low repayment relative to the loan) (`ANNUITY_CREDIT_RATIO`, +0.16) | Long repayment term (low repayment relative to the loan) |

**Applicant 140957**: PD 29.3%, band E (very high), decision **decline** (actual outcome: defaulted).

| Loan officer reasons | Customer reasons |
|---|---|
| Low average score from external credit bureaus (`EXT_SOURCE_MEAN`, +0.59) | Low average score from external credit bureaus |
| Large overdue amounts at other lenders in the past (`BUREAU_MAX_OVERDUE_AMT`, +0.42) | Large overdue amounts at other lenders in the past |
| Low score from external credit bureau (source 2) (`EXT_SOURCE_2`, +0.20) | Low score from external credit bureau (source 2) |
| Low average repayment on previous loans (`PREV_ANNUITY_MEAN`, +0.15) | Low average repayment on previous loans |

**Applicant 290683**: PD 15.0%, band D (high), decision **review** (actual outcome: repaid).

| Loan officer reasons | Customer reasons |
|---|---|
| Low average score from external credit bureaus (`EXT_SOURCE_MEAN`, +0.40) | Low average score from external credit bureaus |
| High share of previous applications refused (`PREV_REFUSED_SHARE`, +0.26) | High share of previous applications refused |
| Previous loans had long terms (`PREV_CNT_PAYMENT_MEAN`, +0.22) | Previous loans had long terms |
| Employer sector (Self-employed) (`ORGANIZATION_TYPE`, +0.16) | Employer sector (Self-employed) |

**Applicant 192711**: PD 15.5%, band D (high), decision **review** (actual outcome: defaulted).

| Loan officer reasons | Customer reasons |
|---|---|
| Low average score from external credit bureaus (`EXT_SOURCE_MEAN`, +0.28) | Low average score from external credit bureaus |
| Short time with current employer (`EMPLOYED_YEARS`, +0.15) | Short time with current employer |
| Long repayment term (low repayment relative to the loan) (`ANNUITY_CREDIT_RATIO`, +0.14) | Long repayment term (low repayment relative to the loan) |
| No score from external credit bureau (source 1) (`EXT_SOURCE_1`, +0.14) | No score from external credit bureau (source 1) |

**Applicant 302776**: PD 4.4%, band B (low), decision **approve** (actual outcome: repaid).

| Loan officer reasons | Customer reasons |
|---|---|
| No score from external credit bureau (source 1) (`EXT_SOURCE_1`, +0.14) | No score from external credit bureau (source 1) |
| Employer sector (Self-employed) (`ORGANIZATION_TYPE`, +0.12) | Employer sector (Self-employed) |
| Work phone number provided (`FLAG_WORK_PHONE`, +0.11) | Work phone number provided |
| High share of previous applications refused (`PREV_REFUSED_SHARE`, +0.10) | High share of previous applications refused |

**Applicant 363831**: PD 3.6%, band B (low), decision **approve** (actual outcome: repaid).

| Loan officer reasons | Customer reasons |
|---|---|
| Low average score from external credit bureaus (`EXT_SOURCE_MEAN`, +0.22) | Low average score from external credit bureaus |
| Low average repayment on previous loans (`PREV_ANNUITY_MEAN`, +0.16) | Low average repayment on previous loans |
| No score from external credit bureau (source 1) (`EXT_SOURCE_1`, +0.11) | No score from external credit bureau (source 1) |
| Loan amount is high relative to the price of the goods (`CREDIT_GOODS_RATIO`, +0.11) | Loan amount is high relative to the price of the goods |

**Applicant 169973**: PD 17.0%, band D (high), decision **review** (actual outcome: repaid).

| Loan officer reasons | Customer reasons |
|---|---|
| Recently opened a loan at another lender (`BUREAU_DAYS_CREDIT_MAX`, +0.21) | Recently opened a loan at another lender |
| Much of the credit at other lenders is still owed (`BUREAU_ACTIVE_DEBT_RATIO`, +0.20) | Much of the credit at other lenders is still owed |
| Open loans at other lenders run for a long time yet (`BUREAU_ACTIVE_ENDDATE_MAX`, +0.17) | Open loans at other lenders run for a long time yet |
| High applicant age (`AGE_YEARS`, +0.14) | Low number of instalments paid on previous loans |
