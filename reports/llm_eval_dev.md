# LLM explanation eval

Backend: `ollama:qwen2.5:3b`, temperature 0. 50 hold-out applicants drawn at random (seed 42) within each decision: 17 declines, 17 reviews and 16 approvals. Each gets an officer and a customer explanation, so 100 explanations in all. Total time 2.7 minutes.

Each draft is checked automatically (`scorecard/narrative.py`). A draft that fails any check is retried once with the failures quoted back to the model; if the retry fails as well, the reader gets a plain template built from the reason codes.

| Check | What it tests |
|---|---|
| decision | The rest of the text doesn't contradict the decision (the opening sentence stating it is added by the code) |
| numbers | Every number is one from the reason codes (or the PD, officer only), and each sentence's numbers come from a single reason |
| factors | No risk factor term (income, employment, debt, ...) that the reason codes given don't mention |
| protected | No gender, marital status, race, religion or nationality terms, and no he/she/his/her |
| officer_only | Customer only: nothing about age, contact details or documents |
| coverage | Every reason's value appears in the text (numbers, or the category for categorical reasons) |
| judgement | No judgement words of the model's own (significant, poor, below average, ...) |
| format | Plain prose: no list markers, no prompt text or retry feedback echoed back |

## Pass rates

| Check | Officer: first draft | Officer: all drafts | Customer: first draft | Customer: all drafts |
|---|---|---|---|---|
| decision | 100% | 100% | 100% | 100% |
| numbers | 96% | 93% | 98% | 95% |
| factors | 98% | 97% | 100% | 100% |
| protected | 100% | 100% | 100% | 100% |
| officer_only | n/a | n/a | 100% | 100% |
| coverage | 92% | 86% | 72% | 62% |
| judgement | 92% | 86% | 98% | 97% |
| format | 98% | 97% | 100% | 94% |
| **all checks** | **84%** | | **72%** | |

"All drafts" counts first drafts and retries together (58 officer drafts, 64 customer drafts).

| | Officer | Customer |
|---|---|---|
| Retried | 16% | 28% |
| Retry fixed it | 0% | 0% |
| Fell back to the template | 16% | 28% |
| Backend errors | 0 | 0 |
| Median seconds per draft | 1.3 | 1.3 |

## Failure examples (44 failed drafts in all)

| Applicant | Reader | Attempt | Failed checks | Draft |
|---|---|---|---|---|
| 299239 (decline) | officer | 1 | factors: factors not in the reason codes: ['income'] | Recommendation: decline. The applicant's average external bureau score is 0.13, in the lowest 10% of applicants. The applicant repays the loan at 8.0% of their yearly income, in the highest 20% of applicants. The identity document was issued 97 days ago, in the highest 10% of applicants. The applicant has had 50.0% of their previous applications refused, in the highest 10% of applicants. |
| 208135 (decline) | customer | 1 | numbers: numbers not in the reason codes: [80.0]; coverage: reason values missing: ['7.9%', '253,642'] | Your application has been declined. Your credit bureau score is 0.30, which is in the lowest 10% of applicants. You repay a larger percentage of your yearly loan than 80% of other applicants. You have the highest overdue amount among 80% of other applicants. Your average credit card balance as a share of the limit is 79.0%, which is in the highest 20% of applicants. |
| 286552 (decline) | customer | 1 | coverage: reason values missing: ['4.9%', 'in 4.3 years'] | Your application has been declined. Your credit bureau score is 0.14, which is in the lowest 10% of applicants. You repay a lower percentage of your yearly loan amount compared to other applicants. Additionally, your average yearly repayment on previous loans is 3,985, which is in the lowest 10% of applicants. Lastly, the time until the last open loan at another lender ends is in the highest 40% of applicants, indicating a longer repayment period. |
| 327934 (decline) | customer | 2 | numbers: numbers mixed from different reasons in: 'Lastly, you work in a self-employed sector, which is shared by 12.4% of applicants."""\nIt broke these rules: coverage: reason values missing: [\'7,288\'].'; format: not plain prose: 'previous draft' | Your application has been declined. Your previous draft was: """Your credit bureau score is 0.16, which is in the lowest 10% of applicants. You repay a smaller percentage of your yearly loan amount (4.7%) compared to other applicants. Additionally, you have a history of repaying an average yearly repayment on previous loans that is in the lowest 30% of applicants. Lastly, you work in a self-employed sector, which is shared by 12.4% of applicants.""" It broke these rules: coverage: reason values missing: ['7,288'].  Your average yearly repayment on previous loans is 7,288, which is in the lowest 30% of applicants. |
| 160362 (decline) | officer | 1 | judgement: judgement words: ['significantly'] | Recommendation: decline. The applicant's employer sector is Construction, shared by 2.2% of applicants, increasing risk. The average of the external bureau scores is 0.39, in the lowest 30% of applicants, raising risk further. The share of instalments paid late in the last year is 11.8%, in the highest 20% of applicants, increasing risk. The average monthly credit card spending and withdrawals is 65,292, in the highest 10% of applicants, significantly increasing risk. |
| 188247 (review) | officer | 1 | coverage: reason values missing: ['2']; judgement: judgement words: ['strongest'] | Recommendation: manual review. The applicant's external bureau score is 0.13, in the lowest 10% of applicants. The average credit card balance as a share of the limit in the last year is 84.5%, in the highest 20% of applicants. The share of instalments paid late in the last year is 31.2%, in the highest 10% of applicants. Two social contacts defaulted (60 days past due), in the highest 10% of applicants. Each of these factors increased the assessed risk, strongest first. |
| 106153 (approve) | customer | 2 | format: not plain prose: 'previous draft' | Your application has been approved. Your previous draft was: """Your average yearly repayment on previous loans is 8,583, which is in the lowest 30% of applicants. Your yearly repayment on this loan is 29,354, which is in the highest 40% of applicants. The time until the last open loan at another lender ends is in the middle of applicants, at 3.6 years. The age of your car is missing for 66.0% of applicants, which increases the assessed risk.""" It broke these rules: coverage: reason values missing: ['not available'].  Your explanation now follows these rules: - Your average yearly repayment on previous loans is 8,583, which is in the lowest 30% of applicants. - Your yearly repayment on this loan is 29,354, which is in the highest 40% of applicants. - The time until the last open loan at another lender ends is in the middle of applicants, at 3.6 years. - The age of your car is missing for 66.0% of applicants, which increases the assessed risk. |
| 241575 (approve) | officer | 1 | numbers: numbers mixed from different reasons in: 'Recommendation: approve\n\nPD: 0.7% (risk band A, very low)\n\nReasons (each one increased the assessed risk, strongest first):\n- Household size: 1 (shared by 22.1% of applicants)\n- Yearly repayment as a share of income: 34.8% (highest 10% of applicants)\n- Instalments paid on previous loans: 11 (lowest 30% of applicants)\n- Amount requested as a multiple of the amount granted (previous loans): 0.96 times (around the middle of applicants)'; judgement: judgement words: ['strongest']; format: not plain prose: 'Reasons (each' | Recommendation: approve. Recommendation: approve  PD: 0.7% (risk band A, very low)  Reasons (each one increased the assessed risk, strongest first): - Household size: 1 (shared by 22.1% of applicants) - Yearly repayment as a share of income: 34.8% (highest 10% of applicants) - Instalments paid on previous loans: 11 (lowest 30% of applicants) - Amount requested as a multiple of the amount granted (previous loans): 0.96 times (around the middle of applicants) |

## Example explanations

**Applicant 304351**: PD 30.6%, band E, decision **decline** (officer text from: llm, customer text from: llm).

| Loan officer | Customer |
|---|---|
| Recommendation: decline. The applicant's average external bureau score is 0.32, in the lowest 20% of applicants. The applicant has 25 instalments still to pay on open loans, in the highest 10% of applicants. The loan amount is 1.40 times the goods price, in the highest 10% of applicants. The applicant has opened 10 loans at other lenders in the last two years, in the highest 10% of applicants. | Your application has been declined. Your credit bureau score is 0.32, which is in the lowest 20% of applicants. You have instalments still to pay on 25 open loans, which is in the highest 10% of applicants. Your loan amount as a multiple of the goods price is 1.40, which is in the highest 10% of applicants. You have opened 10 loans at other lenders in the last two years, which is in the highest 10% of applicants. |

Reason codes given: officer: Average of the external bureau scores: 0.32 (lowest 20% of applicants); Instalments still to pay on open loans: 25 (highest 10% of applicants); Loan amount as a multiple of the goods price: 1.40 times (highest 10% of applicants); Loans opened at other lenders in the last two years: 10 (highest 10% of applicants). Customer: Credit bureau score: 0.32 (lowest 20% of applicants); Instalments still to pay on open loans: 25 (highest 10% of applicants); Loan amount as a multiple of the goods price: 1.40 times (highest 10% of applicants); Loans opened at other lenders in the last two years: 10 (highest 10% of applicants).

**Applicant 113635**: PD 14.7%, band D, decision **review** (officer text from: llm, customer text from: llm).

| Loan officer | Customer |
|---|---|
| Recommendation: manual review. The applicant's average of the external bureau scores is 0.40, in the lowest 30% of applicants. The applicant has a share of previous applications refused of 50.0%, in the highest 10% of applicants. The applicant's employer sector is self-employed, shared by 12.4% of applicants. Home Credit's rating of the home region and city is 3 out of 3, shared by 14.3% of applicants. | Your application has been passed to a loan officer for review. Your credit bureau score is 0.40, which is in the lowest 30% of applicants. You have had 50.0% of your previous applications refused, which is in the highest 10% of applicants. You are self-employed, which is shared by 12.4% of applicants. Your home region rating is 3, which is shared by 14.3% of applicants. |

Reason codes given: officer: Average of the external bureau scores: 0.40 (lowest 30% of applicants); Share of previous applications refused: 50.0% (highest 10% of applicants); Employer sector: Self-employed (shared by 12.4% of applicants); Home Credit's rating of the home region and city (1 best, 3 worst): 3 (shared by 14.3% of applicants). Customer: Credit bureau score: 0.40 (lowest 30% of applicants); Share of previous applications refused: 50.0% (highest 10% of applicants); Employer sector: Self-employed (shared by 12.4% of applicants); Home region rating: 3 (shared by 14.3% of applicants).

## What these results show

**The automated checks are necessary but not enough.** They reliably catch structural problems: a contradicted decision, invented or rounded numbers, a number moved from one reason to another, an off-list factor, protected or officer-only terms, judgement words, and echoed prompt text. They can't catch a misreading that keeps every number in the right place. I read all 78 accepted LLM texts from this run. 8 contain a factual misstatement the checks passed:

| Reader | Accepted LLM texts | With a misstatement | Examples |
|---|---|---|---|
| Officer | 42 | 2 (5%) | "repays a larger proportion of the loan yearly, at 7.9%, compared to the highest 20% of applicants" (garbled comparison); "The applicant has granted total credit by other lenders" |
| Customer | 36 | 6 (17%) | "instalments still to pay on 25 open loans" (25 instalments, not 25 loans, twice); "You have waited 4.7 years until the last open loan ... ends"; "the lowest total credit granted ... among applicants" (it is the lowest 10%); "made instalments on 11 previous loans" with "this does not mitigate the other factors"; "Your home region's population size is 0.03" |

None of the 8 adds a factor, a protected attribute or a wrong decision. They are all misreadings of a reason's meaning. With qwen2.5:3b, roughly 1 in 6 accepted customer explanations says something slightly untrue about the applicant's own data. That's too many to send to customers without review.

**Retries didn't help.** In this run no retry fixed a failed draft: a small model given its own draft and the list of failures mostly repeats it. Every failed draft ended in the template (16% of officer explanations, 28% of customer ones). The fallback works as designed: the reader always gets an accurate text, and the result records where it came from.

**How I got here.** I ran the eval four times and changed the checks and prompts in between, so these numbers are from the final setup, not a blind test:

| Run | Change before the run | Officer: all checks pass first time | Customer: all checks pass first time | Customer fallback |
|---|---|---|---|---|
| 1 | first version | 52% | 14% | 72% |
| 2 | opening sentence added by code, one example in the prompt, coverage and format checks | 88% | 70% | 16% |
| 3 | per-sentence number check, judgement check, echoed-feedback check, retry prompt asks for the text only | 86% | 70% | 28% |
| 4 | clearer customer label for population density | 84% | 72% | 28% |

Run 2's low fallback rate was wrong. On retry the model often echoed my feedback prompt, which contained its old draft, and those texts passed the checks. My manual read found this, and run 3 added the check that catches it. In run 1 the customer drafts mostly dropped the reason values ("your credit bureau score is low"), and the model often buried the decision at the end, so the code now writes the decision sentence itself.

The text isn't fully reproducible. Ollama ran at temperature 0 with a fixed seed, but 18 of the 100 texts differed between runs 3 and 4, including some for applicants whose input didn't change.

**What I'd do with this.**
- For the officer, the LLM text is a readable summary of the reason codes, and the officer sees the codes alongside it.
- For the customer, the template is the safer default with this model: it says exactly what the reason codes say.
- A larger model (the OpenAI or Anthropic backends, not run here because they cost money) might cut the misstatement rate. The same eval and a manual read would show whether it does.
