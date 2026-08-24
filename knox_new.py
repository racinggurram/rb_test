"""Rob Booker Knoxville Divergence: faithful port + Binance USDT-M futures backtest.

Port target is Booker's own MT4 indicator (Knoxville_Divergence_v3.5.mq4, kept alongside
this file), the most complete of the three public implementations. Deltas vs the Pine and
cTrader ports are noted at knox().

Validated against TradingView's built-in "Rob Booker - Knoxville Divergence" (RB_KnoxDiv,
BookerKnoxvilleDivergence@tv-basicstudies) on BINANCE:BTCUSDT.P 4h, Feb-Aug 2026, run with
its own inputs (Bars Back=150, RSI Period=21, Momentum Period=20): every line RB_KnoxDiv
draws is a knox() signal on the exact bar and the exact anchor. The reverse is not
one-for-one, and that is what reads as "extra signals" when the port is checked against a
chart: RB_KnoxDiv draws ONE line per anchor and keeps stretching its right edge while the
rule re-fires off that same anchor, so the 14 firing bars on 2 anchors between 2026-04-15
and 2026-05-06 are 2 lines on screen, and the 8 bars on 2 anchors in 2026-02-01..05 are 2
lines. --one-per-div collapses the stream to that line count. Every public port hardcodes
30 for the window and 30 backtests materially better than 150 (see report.py --lookback),
so 30 stays the default -- but at 30 those long-anchor lines (k=110..123) are missed
entirely, so compare against a chart at 150.
"""
import json, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FAPI = "https://fapi.binance.com/fapi/v1/klines"
CACHE = Path(__file__).parent / "data"
BAR_HOURS = {"1m": 1 / 60, "3m": 0.05, "5m": 1 / 12, "15m": 0.25, "30m": 0.5, "1h": 1.0,
             "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0, "12h": 12.0, "1d": 24.0,
             "3d": 72.0, "1w": 168.0}


# ---------------------------------------------------------------- data
def _klines(symbol: str, interval: str, start: int) -> list:
    rows = []
    while True:
        url = f"{FAPI}?symbol={symbol}&interval={interval}&limit=1500&startTime={start}"
        try:
            batch = json.loads(urllib.request.urlopen(url, timeout=60).read())
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read() or b"{}").get("msg", str(e))
            raise SystemExit(f"binance rejected {symbol} {interval}: {msg}") from None
        rows += batch
        if len(batch) < 1500:
            return rows
        start = batch[-1][0] + 1
        time.sleep(0.35)  # ponytail: keeps request weight ~1500/min, under the 2400 cap


def fetch(symbol: str, interval: str, live: bool = True) -> pd.DataFrame:
    """USDT-M perp history, closed bars only. Cached per symbol+interval; each call tops
    the cache up with whatever bars closed since, so a stale csv cannot silently answer."""
    if interval not in BAR_HOURS:
        raise SystemExit(f"unknown interval {interval!r}; pick from {', '.join(BAR_HOURS)}")
    f = CACHE / f"{symbol}_{interval}.csv"
    old = pd.read_csv(f, parse_dates=["t"]) if f.exists() else None
    if old is not None and "ms" not in old:
        # pandas infers datetime64[s] for date-only strings, so astype("int64") is NOT
        # nanoseconds. Pin the unit or startTime resets to the epoch and history doubles.
        old["ms"] = old.t.astype("datetime64[ms]").astype("int64")
    if old is not None and not live:
        return old.drop(columns="ms")
    start = int(old.ms.iloc[-1]) + 1 if old is not None and len(old) else 0
    now = time.time() * 1000
    new = pd.DataFrame(
        [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]))
         for r in _klines(symbol, interval, start) if r[6] < now],
        columns=["ms", "open", "high", "low", "close"])
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("ms").sort_values("ms").reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no closed bars for {symbol} {interval}")
    df["t"] = pd.to_datetime(df.ms, unit="ms")
    CACHE.mkdir(exist_ok=True)
    df[["ms", "t", "open", "high", "low", "close"]].to_csv(f, index=False)
    return df[["t", "open", "high", "low", "close"]]


# ---------------------------------------------------------------- indicator
def rsi(close: pd.Series, n: int = 21) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _shift(a: np.ndarray, k: int, fill: float) -> np.ndarray:
    """k>0 pulls past values forward (a[t-k]); k<0 pulls future values back (a[t-k])."""
    out = np.full(len(a), fill, dtype=float)
    if abs(k) < len(a):
        if k >= 0:
            out[k:] = a[: len(a) - k]
        else:
            out[:k] = a[-k:]
    return out


def _one_per_anchor(sig: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Keep the first firing bar of each anchor. RB_KnoxDiv plots a divergence into one
    buffer cell per bar, so re-firing off an anchor already drawn just extends that line's
    right edge instead of adding one -- distinct anchors, not firing bars, are what a chart
    shows. Anchors recur out of order (Apr 16 before Apr 15 in the 2026 pile), so this
    tracks the set, not the last one."""
    out = np.zeros(len(sig), bool)
    seen = set()
    for t in np.flatnonzero(sig):
        a = t - k[t]
        if a not in seen:
            seen.add(a)
            out[t] = True
    return out


def knox(df, look_back=30, min_back=4, rsi_n=21, mom_n=20, ob=70.0, os=30.0,
         extreme=3, swing=2, mid=50.0, one_per_div=False):
    """Adds bear/bull signal flags and the anchor distance (bars back) of each.

    A bearish (sell) signal at bar t requires all of:
      RSI[t] >= mid; high[t] is the `extreme`-bar high; and some anchor y = t-k,
      k in [min_back, look_back), with high[y] a (2*swing+1)-bar swing high,
      no high in (t-min_back .. y] above high[t], RSI > ob somewhere in [y, t],
      and momentum[t] <= momentum[y]. Bullish mirrors it. First valid k wins.

    Signal uses only bars <= t-2 for anchor lookups, so it never repaints and has no
    lookahead. one_per_div drops every bar that re-confirms an anchor already signalled,
    leaving the chart's line count. Lij_MC's Pine clone (31mBtCB6) is a different rule
    altogether -- 5-bar fractals of MOMENTUM compared to the previous momentum fractal,
    RSI(17) overbought read only on the detection bar, no window and no price-high filters
    -- so it is no use as a reference: 71 signals to this rule's 863 across five majors on
    4h, and 26 of those 71 have no signal here within 2 bars, i.e. it is not a stricter
    Knoxville but a different one. Transcribed and measured in report.py --variant.
    """
    # Warmup bars hold NaN in the rolling/shifted series and MUST compare False, which is
    # exactly what a NaN comparison yields; errstate only stops numpy narrating it.
    with np.errstate(invalid="ignore"):
        h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
        n = len(df)
        r = rsi(df.close, rsi_n).to_numpy()
        m = (100 * df.close / df.close.shift(mom_n)).to_numpy()

        hi_now = pd.Series(h).rolling(extreme).max().to_numpy() <= h
        lo_now = pd.Series(l).rolling(extreme).min().to_numpy() >= l
        w = 2 * swing + 1
        sw_h = (h >= pd.Series(h).rolling(w, center=True).max().to_numpy()).astype(float)
        sw_l = (l <= pd.Series(l).rolling(w, center=True).min().to_numpy()).astype(float)

        rmax = np.maximum.reduce([_shift(r, j, -np.inf) for j in range(min_back)])
        rmin = np.minimum.reduce([_shift(r, j, np.inf) for j in range(min_back)])
        hmax, lmin = np.full(n, -np.inf), np.full(n, np.inf)
        any_s, any_b = np.zeros(n, bool), np.zeros(n, bool)
        ks, kb = np.zeros(n, int), np.zeros(n, int)

        for k in range(min_back, look_back):
            rk = _shift(r, k, np.nan)
            rmax, rmin = np.maximum(rmax, rk), np.minimum(rmin, rk)
            hmax = np.maximum(hmax, _shift(h, k, -np.inf))
            lmin = np.minimum(lmin, _shift(l, k, np.inf))
            ok_s = (hmax <= h) & (_shift(sw_h, k, 0) > 0.5) & (rmax > ob) & (m <= _shift(m, k, np.inf))
            ok_b = (lmin >= l) & (_shift(sw_l, k, 0) > 0.5) & (rmin < os) & (m >= _shift(m, k, -np.inf))
            ks = np.where(ok_s & ~any_s, k, ks)
            kb = np.where(ok_b & ~any_b, k, kb)
            any_s |= ok_s
            any_b |= ok_b

        df = df.copy()
        df["rsi"], df["mom"] = r, m
        df["bear"] = any_s & hi_now & (r >= mid)
        df["bull"] = any_b & lo_now & (r <= mid)
        if one_per_div:
            df["bear"] = _one_per_anchor(df.bear.to_numpy(), ks)
            df["bull"] = _one_per_anchor(df.bull.to_numpy(), kb)
        df["bear_k"], df["bull_k"] = np.where(df.bear, ks, 0), np.where(df.bull, kb, 0)
    return df


def tabs(df, fast=12, slow=26, kp=70, slowing=10, up=70.0, dn=30.0):
    """Rob Booker Reversal Tabs, same formulas as his MQ4: MACD main line crossing zero
    while the slowed MT4 stochastic sits at an extreme. Booker pairs these with KD."""
    c = df.close
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    ll, hh = df.low.rolling(kp).min(), df.high.rolling(kp).max()
    st = 100 * (c - ll).rolling(slowing).sum() / (hh - ll).rolling(slowing).sum()
    prev = macd.shift()
    return ((prev < 0) & (macd > 0) & (st < dn)).to_numpy(), ((prev > 0) & (macd < 0) & (st > up)).to_numpy()


def confirm(df, window=10):
    """Booker's actual workflow: KD marks the zone, a Reversal Tab triggers the entry.
    Keeps a signal only on bars where a tab fires within `window` bars of a KD."""
    tb, ts = tabs(df)
    near = lambda s: s.astype(float).rolling(window, min_periods=1).max().to_numpy() > 0.5
    df = df.copy()
    df["bear"], df["bull"] = ts & near(df.bear), tb & near(df.bull)
    return df


# ---------------------------------------------------------------- event study
def forward(df, horizons=(1, 5, 10, 20, 40)):
    """Signed forward return from next bar's open, signal vs all-bar baseline."""
    o, c = df.open.to_numpy(), df.close.to_numpy()
    sig = np.where(df.bear, -1, np.where(df.bull, 1, 0))
    rows = []
    for hz in horizons:
        entry, exit_ = _shift(o, -1, np.nan), _shift(c, -hz - 1, np.nan)
        ret = (exit_ / entry - 1) * 100
        for name, mask in (("bear", sig == -1), ("bull", sig == 1)):
            d = ret[mask] * (-1 if name == "bear" else 1)
            d = d[~np.isnan(d)]
            base = ret[~np.isnan(ret)] * (-1 if name == "bear" else 1)
            rows.append(dict(side=name, bars=hz, n=len(d), mean=d.mean() if len(d) else np.nan,
                             win=(d > 0).mean() * 100 if len(d) else np.nan,
                             base_mean=base.mean(), base_win=(base > 0).mean() * 100,
                             t=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan))
    return pd.DataFrame(rows)


def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------------------------------------------------------------- bracket backtest
def backtest(df, r_mult=2.0, max_bars=40, fee=0.0005, carry=0.0, atr_mult=1.0, atr_n=14):
    """Enter next open. Stop is the signal bar's extreme, floored at atr_mult * ATR from
    entry (the raw structure stop sits a few basis points away and is pure noise).
    Target r_mult * risk, time stop at max_bars, stop wins ties inside a bar, one
    position at a time. `carry` = funding per bar held: longs pay, shorts collect."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    bear, bull = df.bear.to_numpy(), df.bull.to_numpy()
    a = atr(df, atr_n).to_numpy()
    t = df.t.to_numpy()
    out, i, n = [], 0, len(df)
    while i < n - 2:
        if not (bear[i] ^ bull[i]):
            i += 1
            continue
        short = bear[i]
        entry = o[i + 1]
        dist = max(abs(h[i] - entry) if short else abs(entry - l[i]), atr_mult * a[i])
        stop = entry + dist if short else entry - dist
        risk = dist
        if risk <= 0:
            i += 1
            continue
        target = entry - r_mult * risk if short else entry + r_mult * risk
        end = min(i + 1 + max_bars, n - 1)
        j, px, why = i + 1, c[end], "time"
        while j <= end:
            if (short and h[j] >= stop) or (not short and l[j] <= stop):
                px, why = stop, "stop"
                break
            if (short and l[j] <= target) or (not short and h[j] >= target):
                px, why = target, "target"
                break
            j += 1
        held = min(j, end) - i
        gross = (entry - px) / entry if short else (px - entry) / entry
        gross += carry * held * (1 if short else -1)
        out.append(dict(t=t[i], side="short" if short else "long", bars=held,
                        R=(gross - 2 * fee) / (risk / entry), why=why))
        i = min(j, end) + 1
    tr = pd.DataFrame(out)
    if tr.empty:
        return tr, {}
    eq = (1 + 0.01 * tr.R).cumprod()
    stats = dict(n=len(tr), win=(tr.R > 0).mean() * 100, expR=tr.R.mean(),
                 t=tr.R.mean() / (tr.R.std(ddof=1) / np.sqrt(len(tr))) if len(tr) > 2 else np.nan,
                 pf=tr.R[tr.R > 0].sum() / -tr.R[tr.R < 0].sum() if (tr.R < 0).any() else np.inf,
                 totR=tr.R.sum(), maxdd=(1 - eq / eq.cummax()).max() * 100,
                 eq=eq.iloc[-1], hold=tr.bars.mean())
    return tr, stats


def control(df, side, n_sig, reps=200, seed=0, eligible=None, **kw):
    """Permutation control: same trade count and same exit rules on random entry bars.
    Separates signal edge from the market drift a long-only rule inherits for free.
    `eligible` restricts the random draw to the same bars a filter allowed, so a regime
    filter cannot take credit for the regime -- only for picking bars inside it."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    col = "bull" if side == "long" else "bear"
    other = "bear" if side == "long" else "bull"
    d[other] = False
    valid = np.arange(60, len(df) - 60)
    if eligible is not None:
        valid = valid[np.asarray(eligible)[valid]]
    out = []
    for _ in range(reps):
        flags = np.zeros(len(df), bool)
        flags[rng.choice(valid, min(n_sig, len(valid)), replace=False)] = True
        d[col] = flags
        _, st = backtest(d, **kw)
        if st:
            out.append(st["expR"])
    return np.array(out)


# ---------------------------------------------------------------- self-check
def _selftest():
    rng = np.random.default_rng(7)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    df = pd.DataFrame(dict(t=pd.date_range("2020", periods=3000, freq="h"),
                           open=px, high=px * 1.004, low=px * 0.996, close=px))
    full = knox(df)
    assert full.bear.any() and full.bull.any(), "no signals on random walk"

    # no lookahead / no repaint: truncating the series must not change past signals
    for cut in (900, 1500, 2400):
        part = knox(df.iloc[:cut])
        assert part.bear.tolist() == full.bear[:cut].tolist(), f"bear repaints at {cut}"
        assert part.bull.tolist() == full.bull[:cut].tolist(), f"bull repaints at {cut}"

    # every bear signal satisfies the documented conditions at its own anchor
    for i in full.index[full.bear]:
        k = full.bear_k[i]
        assert 4 <= k < 30 and full.rsi[i] >= 50
        assert full.high[i] >= full.high[i - 2:i + 1].max()
        assert full.high[i] >= full.high[i - k:i - 3].max()
        assert full.high[i - k] >= full.high[i - k - 2:i - k + 3].max()
        assert full.rsi[i - k:i + 1].max() > 70
        assert full.mom[i] <= full.mom[i - k]

    # one_per_div is a strict subset that keeps exactly one bar per anchor
    one = knox(df, one_per_div=True)
    assert not (one.bear & ~full.bear).any() and not (one.bull & ~full.bull).any()
    for side, kcol in (("bear", "bear_k"), ("bull", "bull_k")):
        idx = one.index[one[side]]
        anchors = [i - one[kcol][i] for i in idx]
        assert len(set(anchors)) == len(anchors), f"{side}: anchor signalled twice"
        assert set(anchors) == {i - full[kcol][i] for i in full.index[full[side]]}, side

    # RSI matches the Wilder recursion it claims to be (pandas seeds on first diff)
    ref = rsi(df.close, 21).to_numpy()
    d = df.close.diff().to_numpy()
    up, dn = max(d[1], 0.0), max(-d[1], 0.0)
    for i in range(2, 300):
        up += (max(d[i], 0) - up) / 21
        dn += (max(-d[i], 0) - dn) / 21
        assert dn > 0 and abs(ref[i] - (100 - 100 / (1 + up / dn))) < 1e-9, i
    # bracket accounting: a bar that gaps through the stop loses exactly -1R minus fees
    d2 = pd.DataFrame(dict(t=pd.date_range("2020", periods=4, freq="h"),
                           open=[100, 100, 100, 100], high=[101, 102, 102, 102],
                           low=[99, 100, 100, 100], close=[100, 101, 101, 101]))
    d2["bear"], d2["bull"] = [True, False, False, False], False
    tr, _ = backtest(d2, r_mult=2, max_bars=2, atr_mult=0)
    assert tr.why[0] == "stop" and abs(tr.R[0] - (-1 - 2 * 0.0005 * 100)) < 1e-9, tr
    print("selftest ok:", int(full.bear.sum()), "bear /", int(full.bull.sum()), "bull on 3k random bars")


# ---------------------------------------------------------------- cli
def run(symbol, interval="4h", start=None, end=None, r_mults=(2.0, 3.0), look_back=30,
        atr_mult=1.0, max_bars=40, fee=0.0005, fund_bp=1.0, risk=0.01, live=True, reps=0,
        long_only=False, one_per_div=False):
    """Backtest one ticker over one date range. Returns (signals, per-side stats, trades)."""
    df = fetch(symbol, interval, live=live)
    # Indicator first, slice second: warming RSI/momentum/anchors up inside the window
    # invents signals along its left edge that no chart shows, and hides the long-anchor
    # ones (k up to look_back) that a chart does show.
    d = knox(df[df.t <= (end or df.t.max())], look_back=look_back, one_per_div=one_per_div)
    d = d[d.t >= (start or df.t.min())].reset_index(drop=True)
    if len(d) < look_back + 80:
        print(f"\n{symbol} {interval}: only {len(d)} bars in range, need "
              f">{look_back + 80} to be worth reading -- skipped")
        return None, pd.DataFrame(), pd.DataFrame()
    carry = fund_bp / 10000 * BAR_HOURS[interval] / 8
    days = (d.t.iloc[-1] - d.t.iloc[0]).days
    print(f"\n{symbol} {interval}  {d.t.iloc[0]:%Y-%m-%d} -> {d.t.iloc[-1]:%Y-%m-%d}  "
          f"{len(d)} bars ({days}d)   signals: {int(d.bull.sum())} bull, {int(d.bear.sum())} bear")
    # ---------------------------------------------------------
    # Print all Knoxville signal dates
    # ---------------------------------------------------------

    print("\n========== BULLISH KNOXVILLE ==========")
    bull = d.loc[d.bull, ["t", "open", "high", "low", "close"]]

    if len(bull):
        print(bull.to_string(index=False))
    else:
        print("None")

    #print("\n========== BEARISH KNOXVILLE ==========")
    #bear = d.loc[d.bear, ["t", "open", "high", "low", "close"]]

    #if len(bear):
    #    print(bear.to_string(index=False))
    #else:
    #    print("None")

    if long_only:
        # Both KD flavours taken LONG: the bull side as a reversal, the bear side inverted
        # because shorting it is the worst trade in the study (see README of findings).
        d = d.assign(bull=(d.bull | d.bear).to_numpy(), bear=False)
        print(f"  long-only mode: {int(d.bull.sum())} long signals, shorts discarded")

    ev = forward(d)
    if ev.n.max():
        print("\n  forward return from next open, % signed with the trade "
              "(base = every bar, same direction)")
        print(ev.rename(columns={"base_mean": "base", "base_win": "base_win%", "win": "win%"})
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rows, all_tr = [], []
    for r in r_mults:
        tr, _ = backtest(d, r_mult=r, max_bars=max_bars, atr_mult=atr_mult, fee=fee, carry=carry)
        if len(tr):
            all_tr.append(tr.assign(sym=symbol, tf=interval, Rt=r))  # Rt = target, R = realised
        for side in ("long", "short"):
            x = tr[tr.side == side] if len(tr) else tr
            if len(x) < 2:
                continue
            eq = (1 + risk * x.R).cumprod()
            row = dict(side=side, R=r, n=len(x), win=100 * (x.R > 0).mean(),
                       expR=x.R.mean(), t=x.R.mean() / (x.R.std(ddof=1) / np.sqrt(len(x))),
                       pf=x.R[x.R > 0].sum() / max(-x.R[x.R < 0].sum(), 1e-9),
                       totR=x.R.sum(), ret=100 * (eq.iloc[-1] - 1),
                       maxdd=100 * (1 - eq / eq.cummax()).max(), hold=x.bars.mean(),
                       thin="thin" if len(x) < 30 else "")
            if reps and len(d) > 200:
                n_sig = int((d.bull if side == "long" else d.bear).sum())
                ctl = control(d, side, n_sig, reps=reps, r_mult=r, max_bars=max_bars,
                              atr_mult=atr_mult, fee=fee, carry=carry)
                row |= dict(ctl=ctl.mean(), pct=100 * (ctl < x.R.mean()).mean())
            rows.append(row)
    res = pd.DataFrame(rows)
    print(f"\n  brackets: stop=max(signal-bar extreme, {atr_mult}*ATR14), target=R*risk, "
          f"{max_bars}-bar cap, {fee:.2%}/side, funding {fund_bp}bp/8h, "
          f"ret/maxdd at {risk:.0%} risk per trade")
    print("  (no trades)" if res.empty else
          res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not res.empty and res.n.max() < 30:
        print("  thin: under 30 trades is not interpretable on its own. Pass several tickers "
              "in one call for the pooled table, or widen with --look-back 150 / a lower --tf.")
    return d, res, (pd.concat(all_tr) if all_tr else pd.DataFrame())


def pooled(trades, risk=0.01):
    """One row per timeframe+side+target across every ticker: the only readable table for
    short histories. `R` is each trade's realised multiple, `Rt` the target it was given.
    Timeframes stay separate -- pooling them would count one move twice."""
    rows = []
    for (tf, side, rt), x in trades.groupby(["tf", "side", "Rt"]):
        x = x.sort_values("t")
        eq = (1 + risk * x.R).cumprod()
        rows.append(dict(tf=tf, side=side, R=rt, syms=x.sym.nunique(), n=len(x),
                         win=100 * (x.R > 0).mean(), expR=x.R.mean(),
                         t=x.R.mean() / (x.R.std(ddof=1) / np.sqrt(len(x))),
                         pf=x.R[x.R > 0].sum() / max(-x.R[x.R < 0].sum(), 1e-9),
                         totR=x.R.sum(), ret=100 * (eq.iloc[-1] - 1),
                         maxdd=100 * (1 - eq / eq.cummax()).max(),
                         syms_pos=int(sum(x.groupby("sym").R.sum() > 0))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Knoxville Divergence backtest on Binance perps")
    p.add_argument("symbols", nargs="*", help="e.g. ASTERUSDT ENAUSDT")
    p.add_argument("--tf", default="4h", help="interval(s), comma separated (default 4h)")
    p.add_argument("--from", dest="start", help="start date, YYYY-MM-DD")
    p.add_argument("--to", dest="end", help="end date, YYYY-MM-DD")
    p.add_argument("--look-back", type=int, default=30, help="divergence window (TV uses 150)")
    p.add_argument("--atr", type=float, default=1.0, help="minimum stop distance in ATR14")
    p.add_argument("--cap", type=int, default=40, help="time stop, bars")
    p.add_argument("--r", default="2,3", help="target R multiples, comma separated")
    p.add_argument("--fee", type=float, default=5.0, help="taker fee per side, bp (default 5)")
    p.add_argument("--fund", type=float, default=1.0, help="funding per 8h, bp (default 1)")
    p.add_argument("--risk", type=float, default=1.0, help="risk per trade, %% (default 1)")
    p.add_argument("--offline", action="store_true", help="use cached bars, skip the top-up")
    p.add_argument("--control", type=int, default=0, metavar="REPS",
                   help="permutation control: expR vs REPS random-entry runs (try 200). "
                        "pct = where the real result lands in that distribution")
    p.add_argument("--long", action="store_true", dest="long_only",
                   help="take EVERY signal long, never short: the only configuration that "
                        "beat a random-entry control on large caps")
    p.add_argument("--one-per-div", action="store_true", dest="one_per_div",
                   help="one signal per anchor instead of one per firing bar: matches the "
                        "line count RB_KnoxDiv draws on a chart (pair with --look-back 150)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest or not a.symbols:
        _selftest()
    book = []
    for tf in a.tf.split(","):
        for sym in a.symbols:
            _, _, tr = run(sym.upper(), tf.strip(), start=a.start, end=a.end,
                           look_back=a.look_back, atr_mult=a.atr, max_bars=a.cap,
                           r_mults=tuple(float(x) for x in a.r.split(",")),
                           fee=a.fee / 10000, fund_bp=a.fund, risk=a.risk / 100,
                           live=not a.offline, reps=a.control, long_only=a.long_only,
                           one_per_div=a.one_per_div)
            if len(tr):
                book.append(tr)
    if book and pd.concat(book).sym.nunique() > 1:
        pool = pooled(pd.concat(book), risk=a.risk / 100)
        print(f"\n=== pooled per timeframe across {pool.syms.max()} tickers "
              f"(syms_pos = tickers with positive total R) ===")
        print(pool.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
