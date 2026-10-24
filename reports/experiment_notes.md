## What these results show

The fold-to-fold standard deviation of AUC is about 0.001, so differences below about 0.002 are noise.

- **The side tables were the biggest gain.** Bureau data added 0.006 AUC to the application-only LightGBM (0.7605 to 0.7666), and the previous applications, instalment, POS and credit card tables added another 0.014 (to 0.7802). Behavioural history mattered more than anything done to the model itself.
- **Tuning added less.** Optuna raised AUC by 0.008 (0.7802 to 0.7882) and KS by 0.015, by choosing more, shallower trees (900 to 1,600 trees, 16 to 27 leaves, learning rate 0.03 to 0.06). It cost about 8 minutes per fold against 4.5 seconds for the default model.
- **Class weights wrecked calibration without improving ranking.** Balanced weights left AUC and PR-AUC unchanged (0.7805 against 0.7802, 0.2681 against 0.2681), but the predicted PDs rose far above actual default rates: ECE went from 0.004 to 0.298 and Brier from 0.067 to 0.177. A PD that feeds an expected-cost decision has to be a real probability, so every later model is unweighted.
- **The scorecard trades about 0.05 AUC for full readability and good calibration.** It scores 0.7395 against 0.7882 for the tuned model, but uses 17 features whose points can be read off a table, and it had the lowest ECE of any model (0.0024). It is a fair challenger model, or the fallback if every decision must be traceable by hand.

## Calibration choice

Platt scaling and isotonic regression were within fold noise of each other and of the uncalibrated tuned model: ECE 0.0036 and 0.0031 against 0.0040, with fold standard deviations of about 0.001. I chose Platt. Isotonic regression is a step function, so it gives many applicants exactly the same PD. Those ties pile up at the decision threshold, where a whole block of applicants would be approved or declined together. Tied applicants with different reason codes would also get identical PDs, so their explanations would look inconsistent. Platt is a smooth monotonic curve that keeps the model's ranking exactly (same AUC, KS and PR-AUC), whereas isotonic's ties cost 0.007 PR-AUC.

## Timing note

The fit times for the two calibrated versions don't include tuning, which was cached from `lgbm_all_tuned` and reused on identical folds. The isotonic run includes the three extra boosters per fold that produce the calibration predictions. The Platt run reused those same boosters, which is why it shows 0.1 s.
