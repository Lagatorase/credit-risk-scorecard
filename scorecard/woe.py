"""Weight-of-evidence binning, information value and the points-based logistic regression scorecard.

WoE here is ln(share of goods / share of bads) in a bin, so a higher WoE means a safer bin.
Everything in this file is fit on the training rows it is given, which inside CV means one training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING = "Missing"
OTHER = "Other"


def _woe_iv(goods: np.ndarray, bads: np.ndarray, smoothing: float = 0.5) -> tuple[np.ndarray, float]:
    # a small smoothing count stops empty bins from giving infinite WoE
    g = (goods + smoothing) / (goods.sum() + smoothing * len(goods))
    b = (bads + smoothing) / (bads.sum() + smoothing * len(bads))
    woe = np.log(g / b)
    return woe, float(np.sum((g - b) * woe))


def _merge(goods: list, bads: list, cuts: list, i: int) -> None:
    # merges bin i with bin i + 1 and removes the boundary between them
    goods[i] += goods.pop(i + 1)
    bads[i] += bads.pop(i + 1)
    cuts.pop(i)


class WoEBinner:
    """Bins every column, then replaces each value with its bin's WoE.

    Numeric columns start from up to 20 quantile bins. Bins under min_bin_share of rows are merged
    into a neighbour, then adjacent bins are merged until the bad rate moves in one direction
    (monotonic bins are easier to explain and less likely to fit noise), then down to max_bins.
    Missing values get their own bin. Categorical columns use one bin per category, with rare
    categories pooled into "Other".
    """

    def __init__(self, max_bins: int = 8, init_bins: int = 20, min_bin_share: float = 0.05,
                 rare_category_share: float = 0.01):
        self.max_bins = max_bins
        self.init_bins = init_bins
        self.min_bin_share = min_bin_share
        self.rare_category_share = rare_category_share

    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        y = np.asarray(y)
        self.bins_ = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.bins_[col] = self._fit_numeric(X[col].to_numpy(dtype="float64"), y)
            else:
                self.bins_[col] = self._fit_categorical(X[col], y)
        self.iv_ = pd.Series({c: b["iv"] for c, b in self.bins_.items()}).sort_values(ascending=False)
        return self

    def _fit_numeric(self, x: np.ndarray, y: np.ndarray) -> dict:
        miss = np.isnan(x)
        xv, yv = x[~miss], y[~miss]
        # interior cut points from quantiles; duplicates collapse for columns with few distinct values
        cuts = list(np.unique(np.quantile(xv, np.linspace(0, 1, self.init_bins + 1)[1:-1]))) if len(xv) else []
        idx = np.searchsorted(cuts, xv, side="right")
        goods = list(np.bincount(idx, weights=1 - yv, minlength=len(cuts) + 1))
        bads = list(np.bincount(idx, weights=yv, minlength=len(cuts) + 1))

        # merge small (or empty) bins into the neighbour with the closer bad rate
        min_rows = self.min_bin_share * len(x)
        while len(goods) > 1:
            sizes = np.maximum(np.add(goods, bads), 1e-9)
            small = np.flatnonzero(sizes < min_rows)
            if not len(small):
                break
            i = int(small[np.argmin(sizes[small])])
            rate = bads[i] / sizes[i]
            if i == 0:
                j = 0
            elif i == len(goods) - 1:
                j = i - 1
            else:
                left = abs(bads[i - 1] / sizes[i - 1] - rate)
                right = abs(bads[i + 1] / sizes[i + 1] - rate)
                j = i - 1 if left <= right else i
            _merge(goods, bads, cuts, j)

        # merge adjacent bins until the bad rate is monotonic in the overall direction
        def rates():
            return np.array(bads) / np.maximum(np.add(goods, bads), 1)

        if len(goods) > 2:
            direction = np.sign(np.corrcoef(np.arange(len(goods)), rates())[0, 1]) or 1.0
            while len(goods) > 1:
                diffs = np.diff(rates()) * direction
                bad = np.flatnonzero(diffs < 0)
                if not len(bad):
                    break
                _merge(goods, bads, cuts, int(bad[0]))

        # finally cap the number of bins by merging the most similar neighbours
        while len(goods) > self.max_bins:
            _merge(goods, bads, cuts, int(np.argmin(np.abs(np.diff(rates())))))

        miss_good, miss_bad = float((1 - y[miss]).sum()), float(y[miss].sum())
        all_goods = np.array(goods + ([miss_good] if miss.any() else []))
        all_bads = np.array(bads + ([miss_bad] if miss.any() else []))
        woe, iv = _woe_iv(all_goods, all_bads)
        n_value_bins = len(goods)
        return {
            "kind": "numeric",
            "cuts": np.array(cuts),
            "woe": woe[:n_value_bins],
            # columns with no missing values in training still need a WoE for missing at scoring time; I use 0 (neutral)
            "woe_missing": float(woe[-1]) if miss.any() else 0.0,
            "iv": iv,
            "counts": all_goods + all_bads,
            "bad_rate": all_bads / np.maximum(all_goods + all_bads, 1),
            "labels": self._numeric_labels(cuts) + ([MISSING] if miss.any() else []),
        }

    @staticmethod
    def _numeric_labels(cuts) -> list[str]:
        # readable labels like "< 0.31", "0.31 to 0.52", ">= 0.52"
        if not len(cuts):
            return ["All values"]
        fmt = lambda v: f"{v:,.4g}"  # noqa: E731
        labels = [f"< {fmt(cuts[0])}"]
        labels += [f"{fmt(a)} to {fmt(b)}" for a, b in zip(cuts[:-1], cuts[1:])]
        labels.append(f">= {fmt(cuts[-1])}")
        return labels

    def _fit_categorical(self, s: pd.Series, y: np.ndarray) -> dict:
        values = s.astype("object").where(s.notna(), MISSING)
        share = values.value_counts(normalize=True)
        # rare categories are pooled so their WoE isn't estimated from a handful of rows
        rare = set(share[share < self.rare_category_share].index) - {MISSING}
        values = values.where(~values.isin(rare), OTHER)
        stats = pd.DataFrame({"g": 1 - y, "b": y}).groupby(values.to_numpy()).sum()
        woe, iv = _woe_iv(stats["g"].to_numpy(float), stats["b"].to_numpy(float))
        counts = (stats["g"] + stats["b"]).to_numpy()
        return {
            "kind": "categorical",
            "map": dict(zip(stats.index, woe)),
            "rare": rare,
            "iv": iv,
            "counts": counts,
            "bad_rate": stats["b"].to_numpy() / counts,
            "labels": list(stats.index),
            "woe": woe,
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        # only the columns passed in, so the scorecard can transform just its selected features
        for col, b in ((c, self.bins_[c]) for c in X.columns if c in self.bins_):
            if b["kind"] == "numeric":
                x = X[col].to_numpy(dtype="float64")
                w = b["woe"][np.searchsorted(b["cuts"], np.nan_to_num(x), side="right")]
                out[col] = np.where(np.isnan(x), b["woe_missing"], w)
            else:
                values = X[col].astype("object").where(X[col].notna(), MISSING)
                values = values.where(~values.isin(b["rare"]), OTHER)
                # categories never seen in training get a neutral WoE of 0
                out[col] = values.map(b["map"]).fillna(b["map"].get(OTHER, 0.0)).to_numpy(dtype="float64")
        return pd.DataFrame(out, index=X.index)

    def bin_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """The label of the bin each value falls into, used by the points table and reason codes."""
        out = {}
        # only the columns passed in, so the scorecard can transform just its selected features
        for col, b in ((c, self.bins_[c]) for c in X.columns if c in self.bins_):
            if b["kind"] == "numeric":
                x = X[col].to_numpy(dtype="float64")
                n_value_bins = len(b["woe"])
                idx = np.searchsorted(b["cuts"], np.nan_to_num(x), side="right")
                labels = np.array(b["labels"][:n_value_bins], dtype=object)[idx]
                out[col] = np.where(np.isnan(x), MISSING, labels)
            else:
                values = X[col].astype("object").where(X[col].notna(), MISSING)
                out[col] = values.where(~values.isin(b["rare"]), OTHER).to_numpy()
        return pd.DataFrame(out, index=X.index)


class LogisticScorecard:
    """WoE binning, then selection by information value, then logistic regression.

    Selection keeps features with IV of at least iv_min, drops any whose WoE is more than corr_max
    correlated with a stronger feature already kept, and drops features whose coefficient has the
    wrong sign (which happens when correlated features fight each other), refitting each time.
    """

    def __init__(self, iv_min: float = 0.02, corr_max: float = 0.7, C: float = 1.0, binner: WoEBinner | None = None):
        self.iv_min = iv_min
        self.corr_max = corr_max
        self.C = C
        self.binner = binner or WoEBinner()

    def fit(self, X: pd.DataFrame, y) -> "LogisticScorecard":
        y = np.asarray(y)
        self.binner.fit(X, y)
        W = self.binner.transform(X)

        # strongest features first, keeping only those not too correlated with one already kept
        candidates = [c for c, iv in self.binner.iv_.items() if iv >= self.iv_min]
        corr = W[candidates].corr().abs()
        kept = []
        for c in candidates:
            if all(corr.loc[c, k] <= self.corr_max for k in kept):
                kept.append(c)

        # with WoE coding every coefficient should be negative (safer bin, lower log-odds of default)
        while True:
            lr = LogisticRegression(C=self.C, max_iter=2000)
            lr.fit(W[kept], y)
            coefs = pd.Series(lr.coef_[0], index=kept)
            if (coefs < 0).all() or len(kept) == 1:
                break
            kept.remove(coefs.idxmax())
        self.features_ = kept
        self.lr_ = lr
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        W = self.binner.transform(X[self.features_])
        return self.lr_.predict_proba(W[self.features_])

    def points_table(self, base_score: float = 600, base_odds: float = 50, pdo: float = 20) -> tuple[float, pd.DataFrame]:
        """Base points and points per bin, on the usual scale: base_score at good:bad odds of base_odds, and pdo points to double the odds.

        Score = offset - factor * log-odds of default, split across features so each bin carries -factor * coef * WoE.
        """
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        base_points = offset - factor * self.lr_.intercept_[0]
        rows = []
        for coef, col in zip(self.lr_.coef_[0], self.features_):
            b = self.binner.bins_[col]
            woes = list(b["woe"]) + ([b["woe_missing"]] if b["kind"] == "numeric" and len(b["labels"]) > len(b["woe"]) else [])
            total = b["counts"].sum()
            for label, w, n, rate in zip(b["labels"], woes, b["counts"], b["bad_rate"]):
                rows.append({"feature": col, "bin": label, "share": n / total, "default_rate": rate,
                             "woe": w, "points": -factor * coef * w})
        return float(base_points), pd.DataFrame(rows)
