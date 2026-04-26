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


def get_latest_signals(df, forward_candles=10, lookback=50, risk_stats=None):
    signals = []
    recent_df = df.tail(lookback).reset_index(drop=True)

    default_stats = {
        "Bullish FVG":   {"tp": 229.98, "sl": 151.78, "prob": "75.36%", "rr": "1.52"},
        "CHoCH Bullish": {"tp": 206.45, "sl": 174.10, "prob": "62.9%",  "rr": "1.19"},
        "BOS Bullish":   {"tp": 196.68, "sl": 190.84, "prob": "64.86%", "rr": "1.03"},
    }

    def get_stats(pattern):
        if risk_stats is not None and pattern in risk_stats:
            return risk_stats[pattern]
        return default_stats.get(pattern, {"tp": 200, "sl": 150, "prob": "N/A", "rr": "N/A"})

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

    return {
        "equal_highs_lows": eq,
        "fvg": fvg,
        "bos_choch": bos,
        "consolidation": con,
        "outcomes": all_outcomes,
        "summary": summary,
        "confluence": confluence
    }
