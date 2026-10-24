# LLM explanation eval

**Final run.** Backend `ollama:qwen2.5:3b`, temperature 0. 50 hold-out applicants drawn at random (seed 2026) within each decision: 17 declines, 17 reviews and 16 approvals, none of them used while tuning. The prompts and checks were frozen before this run, and nothing was changed after it. Each applicant gets an officer and a customer draft, 100 drafts in all, in 2.1 minutes.

In the app and API the officer sees the LLM draft, labelled "AI-drafted summary: check against the reason codes below". The customer gets the plain template by default (`CUSTOMER_EXPLANATION=template`); the customer drafts here measure the LLM option.

Each draft is checked automatically (`scorecard/narrative.py`). A draft that fails any check is replaced by a template built from the reason codes. There is no retry.

| Check | What it tests |
|---|---|
| decision | The text doesn't contradict the decision (the opening sentence stating it is added by the code) |
| numbers | Every number is one from the reason codes (or the PD, officer only), and each sentence's numbers come from a single reason |
| factors | No risk factor term (income, employment, debt, ...) that the reason codes given don't mention |
| protected | No gender, marital status, race, religion or nationality terms, and no he/she/his/her |
| officer_only | Customer only: nothing about age, location, contact details or documents |
| coverage | Every reason's value appears in the text (numbers, or the category for categorical reasons) |
| judgement | No judgement words of the model's own (significant, poor, below average, ...) |
| format | Plain prose: no list markers, no prompt text echoed back |

## Results (final run)

| Check | Officer | Customer |
|---|---|---|
| decision | 100% | 100% |
| numbers | 98% | 96% |
| factors | 100% | 98% |
| protected | 100% | 100% |
| officer_only | n/a | 100% |
| coverage | 88% | 50% |
| judgement | 92% | 98% |
| format | 98% | 100% |
| **all checks pass** | **80%** | **50%** |
| Fell back to the template | 20% | 50% |
| Backend errors | 0 | 0 |
| Median seconds per draft | 1.3 | 1.2 |

## What these results show

**Officer: 80% of drafts pass every check and are shown; the other 20% fall back to the template.** The officer failures are spread across checks: coverage (a reason's value left out, 12%), judgement words (8%), and one draft each for numbers and format. No draft contradicted the decision, named a protected attribute or caused a backend error.

**Customer (the LLM option, off by default): only 50% pass.** Every failed customer draft failed coverage (some failed other checks too): the draft gives a reason's percentile but drops its value ("Your yearly repayment is in the highest 20% of applicants"), or leaves out a category such as "Laborers". This is lower than the 72% of the last tuning run. Two things changed: this is a new sample, and I changed the customer wording of opaque scores (the credit bureau score now shows only its percentile) after the tuning runs. 37 of the 50 customers here had that percentile-only reason, against none in the tuning runs, and the failed drafts often copy its "percentile, no value" pattern onto the other reasons. Within this run, though, drafts with and without that reason pass at a similar rate (19 of 37 against 6 of 13), so with these numbers I can't say how much of the drop it explains. Either way the check caught every case and the customer got the template.

**Manual read: misstatements the checks didn't catch.** I read all 65 accepted LLM texts from this run against their reason codes and counted a misstatement when the text says something about the applicant's data that the reason doesn't (a wrong quantity, a reversed meaning, a current fact turned into a past one or the other way round). Grammar slips that don't change the meaning ("in the shared by 22.1% of applicants") and small losses of detail ("external bureau score" without "(source 3)") aren't counted.

| Reader | Accepted LLM texts | With a misstatement | What went wrong |
|---|---|---|---|
| Officer | 40 | 1 (2.5%) | "Instalments were paid on 9 previous loans" (it is 9 instalments, not 9 loans) |
| Customer | 25 | 5 (20%) | "instalments still to pay on 25 open loans" and the same for 22 (instalments, not loans, twice); "You have no loans at another lender until in 4.6 years" and the same for 4.8 years (the reason says the last open loan ends then, so the text reverses it, twice); "you have an overdue amount of 17,027 at other lenders" (the reason is the largest overdue amount ever, not a current one) |
| Both | 65 | 6 (9%) | |

The pattern matches the tuning runs (officer 2 of 42, customer 6 of 36). The same two misreadings keep coming back: counts of instalments read as counts of loans, and "time until the last open loan ends" read as a wait. Over the 50 applicants, 1 officer explanation in 50 that the app would have shown contains a misstatement; the other 49 are either accurate LLM text or the template. For customers it would be 5 in 50, which is why the customer gets the template by default.

**Reproducibility.** The text isn't fully reproducible. Ollama ran at temperature 0 with a fixed seed (42), but between tuning runs 3 and 4, 18 of the 100 texts differed, including some for applicants whose input didn't change. So a rerun of this eval would give pass rates close to these but not identical, and the same applicant can get a slightly different explanation on different days. The app records the source of every text (LLM, template, or fallback) so this is visible, and the reason codes, which are deterministic, are always shown next to the officer's text.

**What I'd do with this.**
- Keep the officer's LLM text, labelled as an AI draft to check against the reason codes, as it is now.
- Keep the customer on the template. A 20% misstatement rate among accepted customer drafts is too high to send without review.
- A larger model (the OpenAI or Anthropic backends, not run because they cost money) may well do better on both counts. The same script and a manual read would show it.

## Failure examples (35 failed drafts)

| Applicant | Reader | Failed checks | Draft |
|---|---|---|---|
| 450476 (decline) | customer | coverage: reason values missing: ['12.5%', '83.3%', '8'] | Your application has been declined. Your credit bureau score is in the lowest 10% of applicants. You have the highest share of instalments underpaid among applicants. You have the highest share of loans at other lenders still open among applicants. You have the lowest point-of-sale and cash loan history among applicants. |
| 298076 (decline) | officer | judgement: judgement words: ['significantly'] | Recommendation: decline. The applicant's occupation is Low-skill Laborers, shared by 1.0% of applicants, increasing risk. The share of instalments paid late in the last year is 17.6%, in the highest 20% of applicants, significantly increasing risk. The average of the external bureau scores is 0.45, in the lowest 40% of applicants, increasing risk. The total credit granted by other lenders is 55,890, in the lowest 10% of applicants, increasing risk. |
| 373695 (review) | customer | numbers: numbers mixed from different reasons in: 'Your credit bureau score and total credit granted by other lenders are not available, missing for 56.3% and 14.3% of applicants respectively.'; coverage: reason values missing: ['Laborers'] | Your application has been passed to a loan officer for review. Your share of instalments paid late in the last year is 11.1%, which is in the highest 20% of applicants. You are classified as a Laborer, shared by 26.2% of applicants. Your credit bureau score and total credit granted by other lenders are not available, missing for 56.3% and 14.3% of applicants respectively. |
| 296728 (approve) | officer | numbers: numbers mixed from different reasons in: 'Recommendation: approve\n\nPD: 4.4% (risk band B, low)\n\nReasons (each one increased the assessed risk, strongest first):\n- Yearly repayment: 51,543 (highest 10% of applicants)\n- Average term of previous loans: 21 instalments (highest 20% of applicants)\n- External bureau score (source 1): not available (missing for 56.3% of applicants)\n- Amount requested as a multiple of the amount granted (previous loans): 0.88 times (lowest 20% of applicants)'; judgement: judgement words: ['strongest']; format: not plain prose: 'Reasons (each' | Recommendation: approve. Recommendation: approve  PD: 4.4% (risk band B, low)  Reasons (each one increased the assessed risk, strongest first): - Yearly repayment: 51,543 (highest 10% of applicants) - Average term of previous loans: 21 instalments (highest 20% of applicants) - External bureau score (source 1): not available (missing for 56.3% of applicants) - Amount requested as a multiple of the amount granted (previous loans): 0.88 times (lowest 20% of applicants) |
| 199203 (approve) | customer | coverage: reason values missing: ['in 29.6 years', '43,006', 'not available']; judgement: judgement words: ['strong', 'weak'] | Your application has been approved. Your time until the last open loan ends is in the highest 20% of applicants, which is strong. Your yearly repayment amount is in the highest 20% of applicants, which is also strong. However, you have been with your current employer for 1.8 years, which is in the lowest 30% of applicants, which is weak. Your credit bureau score is missing for 56.3% of applicants, which is weak. |
| 384923 (approve) | customer | factors: factors not in the reason codes: ['housing']; coverage: reason values missing: ['Stone, brick'] | Your application has been approved. Your credit grant from other lenders is 0, which is in the lowest 10% of applicants. You repay a yearly amount as a share of your income of 43.2%, which is in the highest 10% of applicants. You repay a yearly amount of 77,782, which is in the highest 10% of applicants. Your property is made of Stone or brick, which is shared by 42.8% of applicants. |
| 358319 (decline) | officer | coverage: reason values missing: ['Business Entity Type 3'] | Recommendation: decline. The applicant's average external bureau score is 0.39, in the lowest 30% of applicants. The loan amount is 1.42 times the goods price, in the highest 10% of applicants. The applicant works in an employer sector shared by 22.1% of applicants. The share of previous loans on higher-rate products is 100.0%, in the highest 10% of applicants. |
| 358319 (decline) | customer | coverage: reason values missing: ['100.0%'] | Your application has been declined. Your credit bureau score is in the lowest 30% of applicants. You have a loan amount that is 1.42 times the price of the goods, which is in the highest 10% of applicants. You work in the Employer sector of Business Entity Type 3, shared by 22.1% of applicants. You have had all previous loans on higher-rate products, which is in the highest 10% of applicants. |

## Example explanations

The officer column is what the app shows. The customer column is the LLM option; by default the customer gets the template shown under it.

**Applicant 282527**: PD 25.4%, band E, decision **decline** (officer text from: llm, customer text from: llm).

| Loan officer | Customer (LLM option) |
|---|---|
| Recommendation: decline. The applicant's average external bureau score is 0.30, in the lowest 10% of applicants. The applicant has an average term of 24 instalments for previous loans, in the highest 20% of applicants. The applicant has 100.0% of loans at other lenders still open, shared by 12.4% of applicants. The applicant has had 75.0% of previous applications refused, in the highest 10% of applicants. | Your application has been declined. Your credit bureau score is in the lowest 10% of applicants. You have had the average term of previous loans of 24 instalments, which is in the highest 20% of applicants. You have 100.0% of your previous loans at other lenders still open, shared by 12.4% of applicants. You have had 75.0% of your previous applications refused, which is in the highest 10% of applicants. |

Customer default (template): Your application has been declined. The factors that weighed most in this assessment were: Credit bureau score: lowest 10% of applicants. Average term of previous loans: 24 instalments (highest 20% of applicants). Share of loans at other lenders still open: 100.0% (shared by 12.4% of applicants). Share of previous applications refused: 75.0% (highest 10% of applicants).

Reason codes given: officer: Average of the external bureau scores: 0.30 (lowest 10% of applicants); Average term of previous loans: 24 instalments (highest 20% of applicants); Share of loans at other lenders still open: 100.0% (shared by 12.4% of applicants); Share of previous applications refused: 75.0% (highest 10% of applicants). Customer: Credit bureau score: lowest 10% of applicants; Average term of previous loans: 24 instalments (highest 20% of applicants); Share of loans at other lenders still open: 100.0% (shared by 12.4% of applicants); Share of previous applications refused: 75.0% (highest 10% of applicants).

**Applicant 152293**: PD 17.8%, band D, decision **review** (officer text from: llm, customer text from: llm).

| Loan officer | Customer (LLM option) |
|---|---|
| Recommendation: manual review. The applicant's average external bureau score is 0.33, in the lowest 20% of applicants. The applicant's average credit card balance as a share of the limit in the last year is 98.3%, in the highest 10% of applicants. The applicant's yearly repayment as a share of the loan is 4.9%, around the middle of applicants. The applicant's average yearly repayment on previous loans is 5,857, in the lowest 20% of applicants. | Your application has been passed to a loan officer for review. Your credit bureau score is in the lowest 20% of applicants. Your average credit card balance as a share of the limit in the last year is 98.3%, which is in the highest 10% of applicants. Your yearly repayment as a share of the loan is 4.9%, which is around the middle of applicants. Your average yearly repayment on previous loans is 5,857, which is in the lowest 20% of applicants. |

Customer default (template): Your application has been passed to a loan officer for review. The factors that weighed most in this assessment were: Credit bureau score: lowest 20% of applicants. Average credit card balance as a share of the limit in the last year: 98.3% (highest 10% of applicants). Yearly repayment as a share of the loan: 4.9% (around the middle of applicants). Average yearly repayment on previous loans: 5,857 (lowest 20% of applicants).

Reason codes given: officer: Average of the external bureau scores: 0.33 (lowest 20% of applicants); Average credit card balance as a share of the limit in the last year: 98.3% (highest 10% of applicants); Yearly repayment as a share of the loan: 4.9% (around the middle of applicants); Average yearly repayment on previous loans: 5,857 (lowest 20% of applicants). Customer: Credit bureau score: lowest 20% of applicants; Average credit card balance as a share of the limit in the last year: 98.3% (highest 10% of applicants); Yearly repayment as a share of the loan: 4.9% (around the middle of applicants); Average yearly repayment on previous loans: 5,857 (lowest 20% of applicants).

## Development results (tuning runs, not the headline)

These come from the runs I used to design the prompts and checks, on a different 50 applicants (seed 42), with a feedback retry that has since been removed. Full detail of the last tuning run is in `reports/llm_eval_dev.md`.

### What these results show

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
