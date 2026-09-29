#!/usr/bin/env python3
"""Repair the 2020-07-06 vendor splice in dataset/full_history price files.

The provider's adjusted-close series consists of two segments adjusted at different
times. Rows before 2020-07-06 are adjusted only for distributions up to that date
(for AT&T the adj/close ratio steps down at each quarterly ex-dividend date and is
exactly 1 from 2020-04-08 to 2020-07-02); rows from 2020-07-06 on are adjusted for
all later distributions. Joining the two vintages leaves a one-day jump that is
not a price move: adj_close falls 42.1% for AT&T, 14.0% for CVX, 9.0% for INTC and
1.3% for BABA, while AMD (no dividends) is unaffected. Across the 4,412 provider
files that contain both days, the median close return on 2020-07-06 is +0.96% but
the median adj_close "return" is -3.02%, and 42.7% of tickers show an adj_close drop
below -5% (4.6% for close): the break is database-wide and exists only in the
adjusted series. AT&T is additionally re-based in open/high/low/close/volume from
2020-07-06 on (a later corporate-action factor applied only to that segment); the
same break appears in the prices released with FNSPID (close 30.08 on 2020-07-02,
23.03 on 2020-07-06), so it comes from the shared upstream source.

Measured in TimeXer's own training scale, AT&T's splice day carries ~5,575x the
squared error of a normal day -- more than its whole training set combined.

The fix rescales every row BEFORE the splice by one constant per column, so the
splice day shows an ordinary return. Returns within each segment are unchanged,
and every row from the splice onward is left byte-identical, so the whole test
period (2022-10 onward) is untouched; only training data changes.

  * a column continuous across the splice is left alone;
  * a spliced price column is multiplied, pre-splice, by
        k = (v_splice / v_prev) / (1 + r_true)
    where r_true is the ticker's own close return that day, or -- when close
    itself is spliced (AT&T only) -- the cross-sectional median close return
    (--market_return, 0.89% over 203 sampled tickers; 0.96% over all 4,412);
  * volume is rescaled (by 1/k) only when close is spliced, keeping dollar volume
    continuous.

Factors applied (pre-2020-07-06 rows): AT&T open/high/low/close x0.7588,
adj_close x0.5739, volume x1.3178; INTC adj_close x0.9034; CVX adj_close x0.8573;
BABA adj_close x0.9867; AMD unchanged. The only assumption is AT&T's return on the
splice day, set to the market median.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SPLICE = "2020-07-06"
PREV = "2020-07-02"          # previous trading day (2020-07-03 was a holiday)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock_dir", default="dataset/full_history")
    ap.add_argument("--output_dir", default="dataset/full_history_fixed")
    ap.add_argument("--tickers", default="T,INTC,AMD,CVX,BABA")
    ap.add_argument("--market_return", type=float, default=0.0089,
                    help="Median close return on the splice day across 203 sampled tickers.")
    ap.add_argument("--close_splice_threshold", type=float, default=0.10,
                    help="A close move beyond this on the splice day is treated as a splice.")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for tk in [t.strip().upper() for t in a.tickers.split(",") if t.strip()]:
        src = Path(a.stock_dir) / f"{tk}.csv"
        with src.open(newline="") as h:
            reader = csv.DictReader(h); fields = reader.fieldnames or []; rows = list(reader)
        col = {f.strip().lower().replace(" ", "_"): f for f in fields}
        by_date = {r[col["date"]][:10]: r for r in rows}
        if SPLICE not in by_date or PREV not in by_date:
            raise SystemExit(f"{tk}: {PREV} or {SPLICE} missing from {src}")
        num = lambda d, c: float(by_date[d][col[c]])
        close_ret = num(SPLICE, "close") / num(PREV, "close") - 1.0
        close_spliced = abs(close_ret) > a.close_splice_threshold
        r_true = a.market_return if close_spliced else close_ret
        factors = {}
        for c in ("open", "high", "low", "close", "adj_close"):
            if c not in col:
                continue
            if c != "adj_close" and not close_spliced:
                continue
            ref = "close" if c in ("open", "high", "low") else c
            factors[c] = (num(SPLICE, ref) / num(PREV, ref)) / (1.0 + r_true)
        if "adj_close" in factors and abs(factors["adj_close"] - 1.0) < 1e-6:
            factors.pop("adj_close")
        vol_factor = (1.0 / factors["close"]) if close_spliced and "volume" in col else None
        fixed = []
        for r in rows:
            r = dict(r)
            if r[col["date"]][:10] < SPLICE:
                for c, k in factors.items():
                    r[col[c]] = repr(float(r[col[c]]) * k)
                if vol_factor is not None and r[col["volume"]] not in ("", None):
                    r[col["volume"]] = repr(float(r[col["volume"]]) * vol_factor)
            fixed.append(r)
        with (out / f"{tk}.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(fixed)
        desc = ", ".join(f"{c} x{k:.4f}" for c, k in factors.items()) or "unchanged"
        if vol_factor: desc += f", volume x{vol_factor:.4f}"
        print(f"{tk:5s} close {'SPLICED' if close_spliced else 'ok'} ({100*close_ret:+.2f}%), "
              f"true return used {100*r_true:+.2f}%  ->  pre-{SPLICE}: {desc}")


if __name__ == "__main__":
    main()
