# Reason codes before and after the rework

The same three hold-out applicants under the first version of the reason codes (commit `2c012be`) and the current one (`610c6f6`). The model, PDs and SHAP values are unchanged; only the wording and selection rules differ.

What changed:

- **Wording.** Each reason now gives the applicant's actual value and where it sits among the 246,008 development applicants, instead of a fixed phrase like "Low ..." or "High ...". The fixed phrases assumed every feature always pushes risk the same way, which a tree model doesn't guarantee.
- **Families.** A list shows at most one reason per family (the strongest by SHAP), so four reasons cover four different things. Before, two reasons could both be bureau scores.
- **Officer-only factors.** Age, social circle, and application, contact and document details only go to the loan officer. Before, only age was held back.
- **Customer labels.** The external scores read "Credit bureau score" for customers, while the officer keeps the specific source.

## Applicant 140957: PD 29.3%, decline (defaulted)

| Before (officer and customer) | After: officer | After: customer |
|---|---|---|
| Low average score from external credit bureaus | Average of the external bureau scores: 0.30 (lowest 10% of applicants) | Credit bureau score: 0.30 (lowest 10% of applicants) |
| Large overdue amounts at other lenders in the past | Largest overdue amount ever at other lenders: 582,898 (highest 10% of applicants) | Largest overdue amount ever at other lenders: 582,898 (highest 10% of applicants) |
| Low score from external credit bureau (source 2) | Average yearly repayment on previous loans: 5,945 (lowest 20% of applicants) | Average yearly repayment on previous loans: 5,945 (lowest 20% of applicants) |
| Low average repayment on previous loans | Occupation: Laborers (shared by 26.2% of applicants) | Occupation: Laborers (shared by 26.2% of applicants) |

The second bureau score is gone (same family as the first), and occupation takes its place.

## Applicant 290683: PD 15.0%, review (repaid)

| Before (officer and customer) | After: officer | After: customer |
|---|---|---|
| Low average score from external credit bureaus | Average of the external bureau scores: 0.40 (lowest 30% of applicants) | Credit bureau score: 0.40 (lowest 30% of applicants) |
| High share of previous applications refused | Share of previous applications refused: 60.0% (highest 10% of applicants) | Share of previous applications refused: 60.0% (highest 10% of applicants) |
| Previous loans had long terms | Average term of previous loans: 30 instalments (highest 10% of applicants) | Average term of previous loans: 30 instalments (highest 10% of applicants) |
| Employer sector (Self-employed) | Employer sector: Self-employed (shared by 12.4% of applicants) | Employer sector: Self-employed (shared by 12.4% of applicants) |

Same four factors; the text now says how far from typical each one is.

## Applicant 169973: PD 17.0%, review (repaid)

| Before: officer | Before: customer | After: officer | After: customer |
|---|---|---|---|
| Recently opened a loan at another lender | Recently opened a loan at another lender | Most recent loan opened at another lender: 22 days ago (highest 10% of applicants) | Most recent loan opened at another lender: 22 days ago (highest 10% of applicants) |
| Much of the credit at other lenders is still owed | Much of the credit at other lenders is still owed | Age: 64.9 years (highest 10% of applicants) | Instalments paid on previous loans: 0 (lowest 10% of applicants) |
| Open loans at other lenders run for a long time yet | Open loans at other lenders run for a long time yet | Instalments paid on previous loans: 0 (lowest 10% of applicants) | Employer sector: Not applicable (shared by 17.9% of applicants) |
| High applicant age | Low number of instalments paid on previous loans | Employer sector: Not applicable (shared by 17.9% of applicants) | Loan amount as a multiple of the goods price: 1.21 times (highest 20% of applicants) |

Before, three of four reasons were about loans at other lenders; now that family appears once. Age stays officer-only. "Not applicable" is Home Credit's `XNA` employer code, mostly pensioners.
