import asyncio
import json
import websockets
import pandas as pd
import plotly.express as px
import streamlit as st
from detector import run_all_detectors, get_latest_signals

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# --- DATA FETCHING ---


async def get_candles(symbol, granularity, count=500):
    async with websockets.connect(DERIV_WS_URL) as ws:
        payload = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles"
        }
        await ws.send(json.dumps(payload))
        response = json.loads(await ws.recv())
        if "error" in response:
            return []
        return response["candles"]


def build_dataframe(candles):
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    df = df[["time", "open", "high", "low", "close"]]
    df[["open", "high", "low", "close"]] = df[[
        "open", "high", "low", "close"]].astype(float)
    df = df.reset_index(drop=True)
    return df


def fetch_data(symbol, granularity, count):
    return asyncio.run(get_candles(symbol, granularity, count))


def render_signal_card(row, direction_color):
    gap_info = ""
    if row.get("pattern") == "Bullish FVG" and row.get("gap_bottom") is not None:
        gap_info = f"""
        <p style="margin:5px 0; color:#aaa">
            📦 FVG Zone: <b style="color:white">{row['gap_bottom']}</b> → <b style="color:white">{row['gap_top']}</b>
            &nbsp;|&nbsp; Midpoint: <b style="color:white">{row['gap_midpoint']}</b>
        </p>
        """

    st.markdown(f"""
    <div style="
        background-color: #1e1e2e;
        border-left: 5px solid {direction_color};
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    ">
        <h4 style="color:{direction_color}; margin:0">
            {row['direction']} — {row['pattern']}
        </h4>
        <p style="color:#aaa; margin:5px 0">
            🕒 Signal confirmed: {row['time']} &nbsp;|&nbsp;
            🎯 Win Probability: <b style="color:white">{row['win_probability']}</b> &nbsp;|&nbsp;
            ⚖️ R:R: <b style="color:white">{row['rr_ratio']}</b>
        </p>
        <p style="margin:5px 0">
            🟡 Entry (candle 4 open): <b style="color:#f0c040">{row['entry']}</b>
        </p>
        <p style="margin:5px 0">
            ✅ TP: <b style="color:#00cc96">{row['suggested_tp']}</b> &nbsp;|&nbsp;
            ❌ SL: <b style="color:#ef553b">{row['suggested_sl']}</b>
        </p>
        {gap_info}
        <p style="color:#555; margin:5px 0; font-size:0.8em">
            ⚠️ Refresh tool to check for newer signals
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_confluence_card(row):
    st.markdown(f"""
    <div style="
        background-color: #1e1e2e;
        border-left: 6px solid {row['color']};
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 6px;
    ">
        <h4 style="color:{row['color']}; margin:0">
            {row['strength']} — Confluence Score: {row['score']}
        </h4>
        <p style="color:#aaa; margin:5px 0">🕒 {row['time']}</p>
        <p style="margin:5px 0; color:#ccc">
            📊 Factors: <b style="color:white">{row['reasons']}</b>
        </p>
        <p style="margin:5px 0">
            🎯 Entry: <b style="color:#f0c040">{row['entry']}</b> &nbsp;|&nbsp;
            ✅ TP: <b style="color:#00cc96">{row['suggested_tp']}</b> &nbsp;|&nbsp;
            ❌ SL: <b style="color:#ef553b">{row['suggested_sl']}</b>
        </p>
        <p style="margin:5px 0">
            📦 FVG Zone: <b style="color:white">{row['gap_bottom']}</b> → <b style="color:white">{row['gap_top']}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- UI ---
st.set_page_config(page_title="Deriv Pattern Tool", layout="wide")
st.title("Deriv Synthetic Indices — Pattern Analyser")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Settings")

symbol = st.sidebar.selectbox("Select Symbol", {
    "Volatility 75": "R_75",
    "Volatility 25": "R_25",
    "Volatility 50": "R_50",
    "Volatility 90": "R_90",
    "Volatility 90 (1s)": "R_90_1S",
    "Volatility 25 (1s)": "R_25_1S",
}.keys())

symbol_map = {
    "Volatility 75": "R_75",
    "Volatility 25": "R_25",
    "Volatility 50": "R_50",
    "Volatility 90": "R_90",
    "Volatility 90 (1s)": "R_90_1S",
    "Volatility 25 (1s)": "R_25_1S",
}

timeframe = st.sidebar.selectbox("Timeframe", {
    "1 Minute": 60,
    "5 Minutes": 300,
    "15 Minutes": 900,
    "1 Hour": 3600,
    "1 Day": 86400
}.keys())

timeframe_map = {
    "1 Minute": 60,
    "5 Minutes": 300,
    "15 Minutes": 900,
    "1 Hour": 3600,
    "1 Day": 86400
}

candle_count = st.sidebar.slider("Candle Count", 100, 5000, 500, 100)
signal_lookback = st.sidebar.slider(
    "Signal Lookback (candles)", 10, 200, 50, 10)
run_button = st.sidebar.button("Run Analysis", use_container_width=True)

# --- MAIN LOGIC ---
if run_button:
    with st.spinner("Fetching data from Deriv..."):
        candles = fetch_data(
            symbol_map[symbol], timeframe_map[timeframe], candle_count)

    if not candles:
        st.error("Failed to fetch data. Check your connection.")
    else:
        df = build_dataframe(candles)
        st.success(f"✅ {len(df)} candles loaded — {symbol} {timeframe}")

        with st.spinner("Running full analysis..."):
            results = run_all_detectors(df)
            signals = get_latest_signals(df, lookback=signal_lookback)
            confluence = results.get("confluence", pd.DataFrame())

        # --- MAIN TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Signals",
            "🎯 Confluence",
            "📊 Pattern Stats",
            "⚖️ Risk Management",
            "🔍 Raw Data"
        ])

        # ── TAB 1: SIGNALS ──
        with tab1:
            st.subheader("Latest Signals")
            st.caption(
                f"Showing top 5 most recent from last {signal_lookback} candles")

            if signals.empty:
                st.info("No signals detected.")
            else:
                # Sort options
                sort_by = st.selectbox(
                    "Sort signals by",
                    ["Most Recent", "Win Probability",
                        "R:R Ratio", "Pattern Type"],
                    key="signal_sort"
                )

                sorted_signals = signals.copy()

                if sort_by == "Most Recent":
                    sorted_signals = sorted_signals.sort_values(
                        "time", ascending=False)
                elif sort_by == "Win Probability":
                    sorted_signals["prob_val"] = sorted_signals["win_probability"].str.replace(
                        "%", "").astype(float)
                    sorted_signals = sorted_signals.sort_values(
                        "prob_val", ascending=False)
                elif sort_by == "R:R Ratio":
                    sorted_signals["rr_val"] = sorted_signals["rr_ratio"].astype(
                        float)
                    sorted_signals = sorted_signals.sort_values(
                        "rr_val", ascending=False)
                elif sort_by == "Pattern Type":
                    sorted_signals = sorted_signals.sort_values("pattern")

                top5 = sorted_signals.head(5)

                for _, row in top5.iterrows():
                    direction_color = "#00cc96" if row["direction"] == "BUY" else "#ef553b"
                    render_signal_card(row, direction_color)

                # Signal summary table below cards
                st.markdown("---")
                st.caption("All signals this session")
                display_cols = ["time", "pattern", "direction", "entry",
                                "suggested_tp", "suggested_sl", "win_probability", "rr_ratio"]
                st.dataframe(signals[display_cols], use_container_width=True)

        # ── TAB 2: CONFLUENCE ──
        with tab2:
            st.subheader("Confluence Scored Setups")
            st.caption(
                "Signals ranked by how many patterns stack at the same level")

            if confluence.empty:
                st.info("No confluence signals found.")
            else:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Setups", len(confluence))
                col_b.metric("High Confluence (7+)",
                             len(confluence[confluence["score"] >= 7]))
                col_c.metric(
                    "Strong (5-6)", len(confluence[confluence["score"].between(5, 6)]))

                st.markdown("---")

                filter_strength = st.selectbox(
                    "Filter by strength",
                    ["All", "🔥 HIGH only", "⚡ STRONG+", "👀 MODERATE+"],
                    key="conf_filter"
                )

                filtered = confluence.copy()
                if filter_strength == "🔥 HIGH only":
                    filtered = filtered[filtered["score"] >= 7]
                elif filter_strength == "⚡ STRONG+":
                    filtered = filtered[filtered["score"] >= 5]
                elif filter_strength == "👀 MODERATE+":
                    filtered = filtered[filtered["score"] >= 3]

                for _, row in filtered.head(10).iterrows():
                    render_confluence_card(row)

                st.markdown("---")
                fig5 = px.histogram(
                    confluence,
                    x="score",
                    nbins=10,
                    title="Confluence Score Distribution",
                    color_discrete_sequence=["#00cc96"]
                )
                fig5.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig5, use_container_width=True, key="fig5")

        # ── TAB 3: PATTERN STATS ──
        with tab3:
            st.subheader("Pattern Frequency & Outcome Probabilities")

            all_patterns = []
            for key, frame in results.items():
                if key in ["outcomes", "summary", "confluence"]:
                    continue
                if not frame.empty and "pattern" in frame.columns:
                    all_patterns.append(frame[["pattern"]])

            if all_patterns:
                combined = pd.concat(all_patterns)
                freq = combined["pattern"].value_counts().reset_index()
                freq.columns = ["Pattern", "Count"]

                col1, col2 = st.columns(2)

                with col1:
                    st.caption("Frequency Table")
                    st.dataframe(freq, use_container_width=True)

                with col2:
                    fig = px.bar(
                        freq,
                        x="Pattern",
                        y="Count",
                        color="Pattern",
                        text="Count",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_tickangle=-30
                    )
                    st.plotly_chart(fig, use_container_width=True, key="fig1")

            st.markdown("---")
            st.caption("Post Pattern Outcome Probabilities")

            if not results["summary"].empty:
                col3, col4 = st.columns(2)

                with col3:
                    st.dataframe(results["summary"], use_container_width=True)

                with col4:
                    fig2 = px.bar(
                        results["summary"],
                        x="pattern",
                        y="probability %",
                        color="outcome",
                        barmode="group",
                        text="probability %",
                        color_discrete_map={
                            "Bullish": "#00cc96",
                            "Bearish": "#ef553b",
                            "Neutral": "#636efa"
                        }
                    )
                    fig2.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_tickangle=-30,
                        legend_title="Outcome"
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="fig2")

            st.markdown("---")
            st.caption("Detailed Pattern Breakdown")
            breakdown_tabs = st.tabs(
                ["Equal Highs/Lows", "Fair Value Gaps", "BOS / CHoCH", "Consolidation"])

            with breakdown_tabs[0]:
                st.dataframe(results["equal_highs_lows"],
                             use_container_width=True)
            with breakdown_tabs[1]:
                st.dataframe(results["fvg"], use_container_width=True)
            with breakdown_tabs[2]:
                st.dataframe(results["bos_choch"], use_container_width=True)
            with breakdown_tabs[3]:
                st.dataframe(results["consolidation"],
                             use_container_width=True)

        # ── TAB 4: RISK MANAGEMENT ──
        with tab4:
            st.subheader("Risk Management")
            st.caption(
                "Average moves, MAE and R:R per pattern — use these to set your TP and SL")

            if not results["outcomes"].empty:
                risk = results["outcomes"].groupby("pattern").agg(
                    avg_up_move=("up_move", "mean"),
                    avg_down_move=("down_move", "mean"),
                    max_up_move=("up_move", "max"),
                    max_down_move=("down_move", "max"),
                    avg_mae=("mae", "mean"),
                    max_mae=("mae", "max"),
                ).reset_index()

                risk["avg_rr_ratio"] = (
                    risk["avg_up_move"] / risk["avg_down_move"]).round(2)
                risk["avg_up_move"] = risk["avg_up_move"].round(5)
                risk["avg_down_move"] = risk["avg_down_move"].round(5)
                risk["max_up_move"] = risk["max_up_move"].round(5)
                risk["max_down_move"] = risk["max_down_move"].round(5)
                risk["avg_mae"] = risk["avg_mae"].round(5)
                risk["max_mae"] = risk["max_mae"].round(5)

                risk.columns = [
                    "Pattern",
                    "Avg Up Move (TP ref)",
                    "Avg Down Move (SL ref)",
                    "Max Up Move",
                    "Max Down Move",
                    "Avg R:R Ratio",
                    "Avg MAE (breathing room)",
                    "Max MAE (worst case)"
                ]

                st.dataframe(risk, use_container_width=True)

                st.markdown("---")

                fig3 = px.bar(
                    risk,
                    x="Pattern",
                    y="Avg R:R Ratio",
                    color="Avg R:R Ratio",
                    text="Avg R:R Ratio",
                    color_continuous_scale="RdYlGn",
                    title="Average R:R Ratio per Pattern"
                )
                fig3.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-30
                )
                st.plotly_chart(fig3, use_container_width=True, key="fig3")

                fig4 = px.bar(
                    risk,
                    x="Pattern",
                    y=["Avg MAE (breathing room)", "Max MAE (worst case)"],
                    barmode="group",
                    title="MAE — Breathing Room Per Pattern",
                    color_discrete_map={
                        "Avg MAE (breathing room)": "#f0c040",
                        "Max MAE (worst case)": "#ef553b"
                    }
                )
                fig4.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-30,
                    legend_title="MAE Type"
                )
                st.plotly_chart(fig4, use_container_width=True, key="fig4")

        # ── TAB 5: RAW DATA ──
        with tab5:
            st.subheader("Raw Candle Data")
            st.caption(f"{len(df)} candles — {symbol} {timeframe}")
            st.dataframe(df, use_container_width=True)
