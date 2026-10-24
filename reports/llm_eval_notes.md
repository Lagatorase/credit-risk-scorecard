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
