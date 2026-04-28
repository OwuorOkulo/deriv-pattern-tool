import pandas as pd
import numpy as np


def detect_equal_highs_lows(df, threshold=0.001):
    results = []
    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, len(df) - 1):
        for j in range(i + 1, min(i + 20, len(df))):
            # Equal Highs
            if abs(highs[i] - highs[j]) / highs[i] < threshold:
                results.append({
                    "pattern": "Equal Highs",
                    "index_1": i,
                    "index_2": j,
                    "time_1": df["time"].iloc[i],
                    "time_2": df["time"].iloc[j],
                    "price": round(highs[i], 5)
                })
            # Equal Lows
            if abs(lows[i] - lows[j]) / lows[i] < threshold:
                results.append({
                    "pattern": "Equal Lows",
                    "index_1": i,
                    "index_2": j,
                    "time_1": df["time"].iloc[i],
                    "time_2": df["time"].iloc[j],
                    "price": round(lows[i], 5)
                })
    return pd.DataFrame(results)


def detect_fvg(df):
    results = []
    for i in range(1, len(df) - 1):
        # Bullish FVG
        if df["low"].iloc[i + 1] > df["high"].iloc[i - 1]:
            results.append({
                "pattern": "Bullish FVG",
                "index": i,
                "time": df["time"].iloc[i],
                "gap_top": df["low"].iloc[i + 1],
                "gap_bottom": df["high"].iloc[i - 1],
                "gap_size": round(df["low"].iloc[i + 1] - df["high"].iloc[i - 1], 5)
            })
        # Bearish FVG
        if df["high"].iloc[i + 1] < df["low"].iloc[i - 1]:
            results.append({
                "pattern": "Bearish FVG",
                "index": i,
                "time": df["time"].iloc[i],
                "gap_top": df["low"].iloc[i - 1],
                "gap_bottom": df["high"].iloc[i + 1],
                "gap_size": round(df["low"].iloc[i - 1] - df["high"].iloc[i + 1], 5)
            })
    return pd.DataFrame(results)


def detect_bos_choch(df):
    results = []
    for i in range(2, len(df) - 1):
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        curr_close = df["close"].iloc[i]
        prev_prev_high = df["high"].iloc[i - 2]
        prev_prev_low = df["low"].iloc[i - 2]

        # BOS Bullish
        if curr_close > prev_high:
            pattern = "CHoCH Bullish" if prev_prev_high > prev_high else "BOS Bullish"
            results.append({
                "pattern": pattern,
                "index": i,
                "time": df["time"].iloc[i],
                "level": round(prev_high, 5)
            })

        # BOS Bearish
        if curr_close < prev_low:
            pattern = "CHoCH Bearish" if prev_prev_low < prev_low else "BOS Bearish"
            results.append({
                "pattern": pattern,
                "index": i,
                "time": df["time"].iloc[i],
                "level": round(prev_low, 5)
            })

    return pd.DataFrame(results)


def detect_consolidation(df, window=10, threshold=0.003):
    results = []
    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        high_range = segment["high"].max()
        low_range = segment["low"].min()
        range_pct = (high_range - low_range) / low_range

        if range_pct < threshold:
            results.append({
                "pattern": "Consolidation",
                "index_start": i - window,
                "index_end": i,
                "time_start": df["time"].iloc[i - window],
                "time_end": df["time"].iloc[i],
                "range_pct": round(range_pct * 100, 4)
            })
    return pd.DataFrame(results)


def analyze_post_pattern_outcome(df, pattern_df, forward_candles=10, index_col="index"):
    outcomes = []

    if pattern_df.empty or index_col not in pattern_df.columns:
        return pd.DataFrame()

    for _, row in pattern_df.iterrows():
        idx = int(row[index_col])

        if idx + forward_candles >= len(df):
            continue

        entry_close = df["close"].iloc[idx]
        future = df.iloc[idx + 1: idx +
                         forward_candles + 1].reset_index(drop=True)

        if future.empty:
            continue

        max_high = future["high"].max()
        min_low = future["low"].min()

        up_move = max_high - entry_close
        down_move = entry_close - min_low

        # MAE calculation
        try:
            peak_idx = future["high"].idxmax()
            trough_idx = future["low"].idxmin()
            mae_bullish = entry_close - future["low"].iloc[:peak_idx + 1].min()
            mae_bearish = future["high"].iloc[:trough_idx +
                                              1].max() - entry_close
        except Exception:
            mae_bullish = 0
            mae_bearish = 0

        if up_move > down_move:
            outcome = "Bullish"
            mae = round(max(mae_bullish, 0), 5)
        elif down_move > up_move:
            outcome = "Bearish"
            mae = round(max(mae_bearish, 0), 5)
        else:
            outcome = "Neutral"
            mae = 0

        outcomes.append({
            "pattern": row["pattern"],
            "outcome": outcome,
            "up_move": round(up_move, 5),
            "down_move": round(down_move, 5),
            "entry_close": round(entry_close, 5),
            "mae": mae
        })

    return pd.DataFrame(outcomes)


def outcome_summary(outcome_df):
    if outcome_df.empty:
        return pd.DataFrame()

    summary = outcome_df.groupby(
        ["pattern", "outcome"]).size().reset_index(name="count")
    total = outcome_df.groupby("pattern").size().reset_index(name="total")
    summary = summary.merge(total, on="pattern")
    summary["probability %"] = (
        summary["count"] / summary["total"] * 100).round(2)
    summary = summary.sort_values(
        ["pattern", "probability %"], ascending=[True, False])

    return summary


def build_risk_stats(outcomes_df):
    if outcomes_df.empty:
        return None

    stats = {}
    grouped = outcomes_df.groupby("pattern")

    for pattern, group in grouped:
        bullish = group[group["outcome"] == "Bullish"]
        total = len(group)
        bull_count = len(bullish)

        avg_up = group["up_move"].mean()
        avg_down = group["down_move"].mean()
        avg_mae = group["mae"].mean()
        rr = round(avg_up / avg_down, 2) if avg_down > 0 else 0
        prob = f"{round(bull_count / total * 100, 2)}%" if total > 0 else "N/A"

        stats[pattern] = {
            "tp": round(avg_up, 5),
            "sl": round(avg_down, 5),
            "mae": round(avg_mae, 5),
            "prob": prob,
            "rr": str(rr)
        }

    return stats


def get_latest_signals(df, forward_candles=10, lookback=50,):
    signals = []
    recent_df = df.tail(lookback).reset_index(drop=True)

    default_stats = {
        "Bullish FVG":   {"tp": 229.98, "sl": 151.78, "prob": "75.36%", "rr": "1.52"},
        "CHoCH Bullish": {"tp": 206.45, "sl": 174.10, "prob": "62.9%",  "rr": "1.19"},
        "BOS Bullish":   {"tp": 196.68, "sl": 190.84, "prob": "64.86%", "rr": "1.03"},
    }

    def get_stats(pattern):
        return default_stats.get(pattern, {"tp": 200, "sl": 150, "mae": 80, "prob": "N/A", "rr": "N/A"})

    # Bullish FVG signals
    # Bullish FVG signals
    for i in range(1, len(recent_df) - 2):
        if recent_df["low"].iloc[i + 1] > recent_df["high"].iloc[i - 1]:
            gap_top = round(recent_df["low"].iloc[i + 1], 5)
            gap_bottom = round(recent_df["high"].iloc[i - 1], 5)
            gap_size = round(gap_top - gap_bottom, 5)
            gap_midpoint = round((gap_top + gap_bottom) / 2, 5)

            # Entry at open of candle 4 (i+2)
            entry = round(recent_df["open"].iloc[i + 2], 5)

            s = get_stats("Bullish FVG")

            # SL below gap bottom with MAE buffer
            mae_buffer = s.get("mae", 0)
            sl = round(gap_bottom - mae_buffer, 5)

            # TP based on average up move from entry
            tp = round(entry + s["tp"], 5)

            # R:R recalculated from actual entry
            risk = round(entry - sl, 5)
            reward = round(tp - entry, 5)
            rr = round(reward / risk, 2) if risk > 0 else 0

            # Only add if SL is below entry and TP is above entry
            if sl < entry < tp:
                signals.append({
                    "time": recent_df["time"].iloc[i + 2],
                    "pattern": "Bullish FVG",
                    "direction": "BUY",
                    "entry": entry,
                    "gap_top": gap_top,
                    "gap_bottom": gap_bottom,
                    "gap_size": gap_size,
                    "gap_midpoint": gap_midpoint,
                    "suggested_tp": tp,
                    "suggested_sl": sl,
                    "win_probability": s["prob"],
                    "rr_ratio": str(rr)
                })
    # CHoCH Bullish signals
    for i in range(2, len(recent_df) - 1):
        prev_high = recent_df["high"].iloc[i - 1]
        prev_prev_high = recent_df["high"].iloc[i - 2]
        curr_close = recent_df["close"].iloc[i]

        if curr_close > prev_high and prev_prev_high > prev_high:
            entry = round(curr_close, 5)
            s = get_stats("CHoCH Bullish")
            signals.append({
                "time": recent_df["time"].iloc[i],
                "pattern": "CHoCH Bullish",
                "direction": "BUY",
                "entry": entry,
                "gap_top": None,
                "gap_bottom": None,
                "gap_size": None,
                "suggested_tp": round(entry + s["tp"], 5),
                "suggested_sl": round(entry - s["sl"], 5),
                "win_probability": s["prob"],
                "rr_ratio": s["rr"]
            })

    # BOS Bullish signals
    for i in range(2, len(recent_df) - 1):
        prev_high = recent_df["high"].iloc[i - 1]
        prev_prev_high = recent_df["high"].iloc[i - 2]
        curr_close = recent_df["close"].iloc[i]

        if curr_close > prev_high and prev_prev_high <= prev_high:
            entry = round(curr_close, 5)
            s = get_stats("BOS Bullish")
            signals.append({
                "time": recent_df["time"].iloc[i],
                "pattern": "BOS Bullish",
                "direction": "BUY",
                "entry": entry,
                "gap_top": None,
                "gap_bottom": None,
                "gap_size": None,
                "suggested_tp": round(entry + s["tp"], 5),
                "suggested_sl": round(entry - s["sl"], 5),
                "win_probability": s["prob"],
                "rr_ratio": s["rr"]
            })

    if signals:
        sig_df = pd.DataFrame(signals)
        sig_df = sig_df.sort_values(
            "time", ascending=False).reset_index(drop=True)

        # Safety filter — remove bad signals
        sig_df = sig_df[
            ~((sig_df["direction"] == "BUY") & (
                sig_df["suggested_sl"] >= sig_df["entry"]))
        ]
        sig_df = sig_df[
            ~((sig_df["direction"] == "BUY") & (
                sig_df["suggested_tp"] <= sig_df["entry"]))
        ]

        return sig_df.reset_index(drop=True)

    return pd.DataFrame()


def calculate_confluence(df, results, price_tolerance=0.002):
    signals = []

    fvg = results.get("fvg", pd.DataFrame())
    bos = results.get("bos_choch", pd.DataFrame())
    eq = results.get("equal_highs_lows", pd.DataFrame())
    con = results.get("consolidation", pd.DataFrame())

    bullish_fvg = fvg[fvg["pattern"] ==
                      "Bullish FVG"] if not fvg.empty else pd.DataFrame()

    for _, fvg_row in bullish_fvg.iterrows():
        score = 0
        reasons = []
        anchor_price = fvg_row["gap_bottom"]
        anchor_idx = fvg_row["index"]

        score += 3
        reasons.append("Bullish FVG (+3)")

        if not bos.empty:
            choch = bos[
                (bos["pattern"] == "CHoCH Bullish") &
                (abs(bos["index"] - anchor_idx) <= 20)
            ]
            if not choch.empty:
                score += 3
                reasons.append("CHoCH Bullish (+3)")

            bos_bull = bos[
                (bos["pattern"] == "BOS Bullish") &
                (abs(bos["index"] - anchor_idx) <= 20)
            ]
            if not bos_bull.empty:
                score += 2
                reasons.append("BOS Bullish (+2)")

        if not eq.empty:
            eq_lows = eq[
                (eq["pattern"] == "Equal Lows") &
                (abs(eq["price"] - anchor_price) /
                 anchor_price < price_tolerance)
            ]
            if not eq_lows.empty:
                score += 2
                reasons.append("Equal Lows nearby (+2)")

        if not con.empty:
            con_before = con[
                (con["index_end"] >= anchor_idx - 15) &
                (con["index_end"] <= anchor_idx)
            ]
            if not con_before.empty:
                score += 1
                reasons.append("Prior Consolidation (+1)")

        if score >= 6:
            strength = "🔥 HIGH"
            color = "#00cc96"
        elif score >= 4:
            strength = "⚡ STRONG"
            color = "#f0c040"
        elif score >= 3:
            strength = "👀 MODERATE"
            color = "#636efa"
        else:
            strength = "⚠️ WEAK"
            color = "#ef553b"

        entry = round((fvg_row["gap_top"] + fvg_row["gap_bottom"]) / 2, 5)

        signals.append({
            "time": fvg_row["time"],
            "index": anchor_idx,
            "entry": entry,
            "gap_top": round(fvg_row["gap_top"], 5),
            "gap_bottom": round(fvg_row["gap_bottom"], 5),
            "score": score,
            "strength": strength,
            "color": color,
            "reasons": " | ".join(reasons),
            "suggested_tp": round(entry + 229.98, 5),
            "suggested_sl": round(entry - 151.78, 5),
        })

    if signals:
        sig_df = pd.DataFrame(signals)
        sig_df = sig_df.sort_values(
            "score", ascending=False).reset_index(drop=True)
        return sig_df

    return pd.DataFrame()

def analyze_eq_ict_sequence(df, forward_candles=30, threshold=0.001):
    results = []

    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, len(df) - forward_candles - 2):
        for j in range(i + 1, min(i + 20, len(df) - forward_candles - 2)):

            # --- EQUAL LOWS SEQUENCE ---
            if abs(lows[i] - lows[j]) / lows[i] < threshold:
                eq_low_price = lows[j]
                future = df.iloc[j + 1: j + forward_candles + 1].reset_index(drop=True)

                if future.empty:
                    continue

                # Step 1 — Did price sweep below equal low?
                sweep = future["low"].min() < eq_low_price
                sweep_candle = None
                if sweep:
                    for k in range(len(future)):
                        if future["low"].iloc[k] < eq_low_price:
                            sweep_candle = k + 1
                            break

                # Step 2 — After sweep, did a Bullish FVG form?
                fvg_after_sweep = False
                if sweep and sweep_candle is not None:
                    post_sweep = future.iloc[sweep_candle:].reset_index(drop=True)
                    for k in range(1, len(post_sweep) - 1):
                        if post_sweep["low"].iloc[k + 1] > post_sweep["high"].iloc[k - 1]:
                            fvg_after_sweep = True
                            break

                # Step 3 — Did price continue upward after sequence?
                continued_up = False
                if fvg_after_sweep:
                    post_sweep_high = future.iloc[sweep_candle:]["high"].max() if sweep_candle else 0
                    continued_up = post_sweep_high > eq_low_price

                # Max move up and down after equal low
                max_up = round(future["high"].max() - eq_low_price, 5)
                max_down = round(eq_low_price - future["low"].min(), 5)

                results.append({
                    "type": "Equal Lows",
                    "time": df["time"].iloc[j],
                    "price": round(eq_low_price, 5),
                    "sweep_occurred": sweep,
                    "sweep_candle": sweep_candle,
                    "fvg_after_sweep": fvg_after_sweep,
                    "continued_up": continued_up,
                    "full_sequence": sweep and fvg_after_sweep and continued_up,
                    "max_up_after": max_up,
                    "max_down_after": max_down,
                })

            # --- EQUAL HIGHS SEQUENCE ---
            if abs(highs[i] - highs[j]) / highs[i] < threshold:
                eq_high_price = highs[j]
                future = df.iloc[j + 1: j + forward_candles + 1].reset_index(drop=True)

                if future.empty:
                    continue

                # Step 1 — Did price sweep above equal high?
                sweep = future["high"].max() > eq_high_price
                sweep_candle = None
                if sweep:
                    for k in range(len(future)):
                        if future["high"].iloc[k] > eq_high_price:
                            sweep_candle = k + 1
                            break

                # Step 2 — After sweep, did a Bearish FVG form?
                fvg_after_sweep = False
                if sweep and sweep_candle is not None:
                    post_sweep = future.iloc[sweep_candle:].reset_index(drop=True)
                    for k in range(1, len(post_sweep) - 1):
                        if post_sweep["high"].iloc[k + 1] < post_sweep["low"].iloc[k - 1]:
                            fvg_after_sweep = True
                            break

                # Step 3 — Did price continue downward after sequence?
                continued_down = False
                if fvg_after_sweep:
                    post_sweep_low = future.iloc[sweep_candle:]["low"].min() if sweep_candle else 0
                    continued_down = post_sweep_low < eq_high_price

                max_up = round(future["high"].max() - eq_high_price, 5)
                max_down = round(eq_high_price - future["low"].min(), 5)

                results.append({
                    "type": "Equal Highs",
                    "time": df["time"].iloc[j],
                    "price": round(eq_high_price, 5),
                    "sweep_occurred": sweep,
                    "sweep_candle": sweep_candle,
                    "fvg_after_sweep": fvg_after_sweep,
                    "continued_down": continued_up if 'continued_up' in dir() else continued_down,
                    "full_sequence": sweep and fvg_after_sweep and continued_down,
                    "max_up_after": max_up,
                    "max_down_after": max_down,
                })

    return pd.DataFrame(results)


def summarize_eq_ict_sequence(eq_df):
    if eq_df.empty:
        return {}

    summary = {}

    for eq_type in ["Equal Lows", "Equal Highs"]:
        subset = eq_df[eq_df["type"] == eq_type]
        if subset.empty:
            continue

        total = len(subset)
        sweeps = subset["sweep_occurred"].sum()
        fvg_after = subset["fvg_after_sweep"].sum()
        full_seq = subset["full_sequence"].sum()

        sweep_rate = round(sweeps / total * 100, 2)
        fvg_rate = round(fvg_after / sweeps * 100, 2) if sweeps > 0 else 0
        full_seq_rate = round(full_seq / total * 100, 2)

        avg_up = round(subset["max_up_after"].mean(), 5)
        avg_down = round(subset["max_down_after"].mean(), 5)
        max_up = round(subset["max_up_after"].max(), 5)
        max_down = round(subset["max_down_after"].max(), 5)

        summary[eq_type] = {
            "total": total,
            "sweep_count": int(sweeps),
            "sweep_rate": sweep_rate,
            "fvg_after_sweep_count": int(fvg_after),
            "fvg_after_sweep_rate": fvg_rate,
            "full_sequence_count": int(full_seq),
            "full_sequence_rate": full_seq_rate,
            "avg_up_after": avg_up,
            "avg_down_after": avg_down,
            "max_up_after": max_up,
            "max_down_after": max_down,
        }

    return summary

def run_all_detectors(df, forward_candles=10):
    print("\n--- Running Pattern Detection ---\n")

    eq = detect_equal_highs_lows(df)
    fvg = detect_fvg(df)
    bos = detect_bos_choch(df)
    con = detect_consolidation(df)

    print(f"Equal Highs/Lows found: {len(eq)}")
    print(f"Fair Value Gaps found:  {len(fvg)}")
    print(f"BOS/CHoCH found:        {len(bos)}")
    print(f"Consolidation zones:    {len(con)}")

    eq_outcomes = analyze_post_pattern_outcome(
        df, eq, forward_candles, index_col="index_1")
    fvg_outcomes = analyze_post_pattern_outcome(
        df, fvg, forward_candles, index_col="index")
    bos_outcomes = analyze_post_pattern_outcome(
        df, bos, forward_candles, index_col="index")
    con_outcomes = analyze_post_pattern_outcome(
        df, con, forward_candles, index_col="index_start")

    all_outcomes = pd.concat(
        [eq_outcomes, fvg_outcomes, bos_outcomes, con_outcomes])
    summary = outcome_summary(all_outcomes)

    confluence = calculate_confluence(df, {
        "fvg": fvg,
        "bos_choch": bos,
        "equal_highs_lows": eq,
        "consolidation": con
    })
    fvg_behaviour = analyze_fvg_behaviour(df)
    fvg_summary = summarize_fvg_behaviour(fvg_behaviour)
    eq_ict = analyze_eq_ict_sequence(df)
    eq_ict_summary = summarize_eq_ict_sequence(eq_ict)

    return {
        "equal_highs_lows": eq,
        "fvg": fvg,
        "bos_choch": bos,
        "consolidation": con,
        "outcomes": all_outcomes,
        "summary": summary,
        "confluence": confluence,
        "fvg_behaviour": fvg_behaviour,
        "fvg_summary": fvg_summary,
        "eq_ict": eq_ict,
        "eq_ict_summary": eq_ict_summary,
    }


def analyze_fvg_behaviour(df, forward_candles=30):
    fvg_list = []

    for i in range(1, len(df) - forward_candles - 2):
        # Detect Bullish FVG
        if df["low"].iloc[i + 1] > df["high"].iloc[i - 1]:
            gap_top = df["low"].iloc[i + 1]
            gap_bottom = df["high"].iloc[i - 1]
            gap_mid = (gap_top + gap_bottom) / 2

            # Entry at candle 4 open
            entry_idx = i + 2
            entry_price = df["open"].iloc[entry_idx]

            future = df.iloc[entry_idx + 1: entry_idx +
                             forward_candles + 1].reset_index(drop=True)

            if future.empty:
                continue

            # --- FAILURE ANALYSIS ---
            # Failure = price came down into FVG and traded below gap_bottom
            failure = False
            failure_candle = None
            for j in range(len(future)):
                if future["low"].iloc[j] < gap_bottom:
                    failure = True
                    failure_candle = j + 1
                    break

            # --- DRAWDOWN ANALYSIS ---
            # How far did price drop from entry before recovering
            # Look at lowest point before the highest point
            peak_idx = future["high"].idxmax()
            pre_peak = future.iloc[:peak_idx + 1]
            drawdown = entry_price - pre_peak["low"].min()
            drawdown = round(max(drawdown, 0), 5)

            # --- TP RANGE ANALYSIS ---
            # How many candles did price go up before retracing back below entry
            tp_candles = 0
            max_up = 0
            for j in range(len(future)):
                if future["close"].iloc[j] > entry_price:
                    tp_candles = j + 1
                    max_up = max(max_up, future["high"].iloc[j] - entry_price)
                else:
                    break

            # Max up move overall
            max_up_move = round(future["high"].max() - entry_price, 5)

            # Candles before price retraces below entry
            candles_up = 0
            for j in range(len(future)):
                if future["low"].iloc[j] >= entry_price:
                    candles_up = j + 1
                else:
                    break

            fvg_list.append({
                "time": df["time"].iloc[i],
                "entry": round(entry_price, 5),
                "gap_top": round(gap_top, 5),
                "gap_bottom": round(gap_bottom, 5),
                "gap_size": round(gap_top - gap_bottom, 5),
                "failure": failure,
                "failure_candle": failure_candle,
                "drawdown": drawdown,
                "max_up_move": max_up_move,
                "candles_up_before_retrace": candles_up,
            })

    if not fvg_list:
        return pd.DataFrame()

    result_df = pd.DataFrame(fvg_list)

    return result_df


def summarize_fvg_behaviour(fvg_df):
    if fvg_df.empty:
        return {}

    total = len(fvg_df)
    failures = fvg_df[fvg_df["failure"] == True]
    successes = fvg_df[fvg_df["failure"] == False]

    # Failure stats
    failure_count = len(failures)
    failure_rate = round(failure_count / total * 100, 2)
    avg_failure_candle = round(
        failures["failure_candle"].mean(), 2) if not failures.empty else 0

    # Drawdown stats — on successful trades
    avg_drawdown = round(fvg_df["drawdown"].mean(), 5)
    max_drawdown = round(fvg_df["drawdown"].max(), 5)
    drawdown_over_50 = len(
        fvg_df[fvg_df["drawdown"] > fvg_df["drawdown"].mean()])

    # TP range stats
    avg_up_move = round(fvg_df["max_up_move"].mean(), 5)
    max_up_move = round(fvg_df["max_up_move"].max(), 5)
    avg_candles_up = round(fvg_df["candles_up_before_retrace"].mean(), 2)
    max_candles_up = int(fvg_df["candles_up_before_retrace"].max())

    # Best TP buckets — how many times did price go up by various ranges
    buckets = {
        "Moved 0-100 pts": len(fvg_df[fvg_df["max_up_move"] <= 100]),
        "Moved 100-200 pts": len(fvg_df[(fvg_df["max_up_move"] > 100) & (fvg_df["max_up_move"] <= 200)]),
        "Moved 200-300 pts": len(fvg_df[(fvg_df["max_up_move"] > 200) & (fvg_df["max_up_move"] <= 300)]),
        "Moved 300-500 pts": len(fvg_df[(fvg_df["max_up_move"] > 300) & (fvg_df["max_up_move"] <= 500)]),
        "Moved 500+ pts": len(fvg_df[fvg_df["max_up_move"] > 500]),
    }

    return {
        "total_fvgs": total,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "avg_failure_candle": avg_failure_candle,
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
        "drawdown_over_avg": drawdown_over_50,
        "avg_up_move": avg_up_move,
        "max_up_move": max_up_move,
        "avg_candles_up": avg_candles_up,
        "max_candles_up": max_candles_up,
        "tp_buckets": buckets,
    }
