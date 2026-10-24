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
