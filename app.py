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
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.reset_index(drop=True)
    return df

def fetch_data(symbol, granularity, count):
    return asyncio.run(get_candles(symbol, granularity, count))

def render_signal_card(row, direction_color):
    pattern = row["pattern"]
    direction = row["direction"]
    time = row["time"]
    win_prob = row["win_probability"]
    rr = row["rr_ratio"]
    entry = row["entry"]
    tp = row["suggested_tp"]
    sl = row["suggested_sl"]

    fvg_line = ""
    if pattern == "Bullish FVG":
        try:
            gb = row["gap_bottom"]
            gt = row["gap_top"]
            gm = row["gap_midpoint"]
            if pd.notna(gb):
                fvg_line = f'<div style="margin:5px 0; color:#aaa">📦 FVG Zone: <b style="color:white">{gb}</b> to <b style="color:white">{gt}</b> | Midpoint: <b style="color:white">{gm}</b></div>'
        except Exception:
            fvg_line = ""

    html = f"""
    <div style="background-color:#1e1e2e; border-left:5px solid {direction_color}; padding:15px; margin-bottom:10px; border-radius:5px;">
        <h4 style="color:{direction_color}; margin:0">{direction} — {pattern}</h4>
        <div style="color:#aaa; margin:5px 0">🕒 {time} &nbsp;|&nbsp; 🎯 Win Probability: <b style="color:white">{win_prob}</b> &nbsp;|&nbsp; ⚖️ R:R: <b style="color:white">{rr}</b></div>
        <div style="margin:5px 0">🟡 Entry (candle 4 open): <b style="color:#f0c040">{entry}</b></div>
        <div style="margin:5px 0">✅ TP: <b style="color:#00cc96">{tp}</b> &nbsp;|&nbsp; ❌ SL: <b style="color:#ef553b">{sl}</b></div>
        {fvg_line}
        <div style="color:#555; margin-top:8px; font-size:0.8em">⚠️ Refresh tool to check for newer signals</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_confluence_card(row):
    html = f"""
    <div style="background-color:#1e1e2e; border-left:6px solid {row['color']}; padding:15px; margin-bottom:12px; border-radius:6px;">
        <h4 style="color:{row['color']}; margin:0">{row['strength']} — Confluence Score: {row['score']}</h4>
        <div style="color:#aaa; margin:5px 0">🕒 {row['time']}</div>
        <div style="margin:5px 0; color:#ccc">📊 Factors: <b style="color:white">{row['reasons']}</b></div>
        <div style="margin:5px 0">🎯 Entry: <b style="color:#f0c040">{row['entry']}</b> &nbsp;|&nbsp; ✅ TP: <b style="color:#00cc96">{row['suggested_tp']}</b> &nbsp;|&nbsp; ❌ SL: <b style="color:#ef553b">{row['suggested_sl']}</b></div>
        <div style="margin:5px 0">📦 FVG Zone: <b style="color:white">{row['gap_bottom']}</b> to <b style="color:white">{row['gap_top']}</b></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

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
signal_lookback = st.sidebar.slider("Signal Lookback (candles)", 10, 200, 50, 10)
run_button = st.sidebar.button("Run Analysis", use_container_width=True)

# --- MAIN LOGIC ---
if run_button:
    with st.spinner("Fetching data from Deriv..."):
        candles = fetch_data(symbol_map[symbol], timeframe_map[timeframe], candle_count)

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🚨 Signals",
            "🎯 Confluence",
            "📊 Pattern Stats",
            "⚖️ Risk Management",
            "🔬 FVG Deep Analysis",
            "🔍 Raw Data"
        ])

        # ── TAB 1: SIGNALS ──
        with tab1:
            st.subheader("Latest Signals")
            st.caption(f"Showing top 5 most recent from last {signal_lookback} candles")

            if signals.empty:
                st.info("No signals detected.")
            else:
                sort_by = st.selectbox(
                    "Sort signals by",
                    ["Most Recent", "Win Probability", "R:R Ratio", "Pattern Type"],
                    key="signal_sort"
                )

                sorted_signals = signals.copy()

                if sort_by == "Most Recent":
                    sorted_signals = sorted_signals.sort_values("time", ascending=False)
                elif sort_by == "Win Probability":
                    sorted_signals["prob_val"] = sorted_signals["win_probability"].str.replace("%", "").astype(float)
                    sorted_signals = sorted_signals.sort_values("prob_val", ascending=False)
                elif sort_by == "R:R Ratio":
                    sorted_signals["rr_val"] = sorted_signals["rr_ratio"].astype(float)
                    sorted_signals = sorted_signals.sort_values("rr_val", ascending=False)
                elif sort_by == "Pattern Type":
                    sorted_signals = sorted_signals.sort_values("pattern")

                top5 = sorted_signals.head(5)

                for _, row in top5.iterrows():
                    direction_color = "#00cc96" if row["direction"] == "BUY" else "#ef553b"
                    render_signal_card(row, direction_color)

                st.markdown("---")
                st.caption("All signals this session")
                display_cols = ["time", "pattern", "direction", "entry", "suggested_tp", "suggested_sl", "win_probability", "rr_ratio"]
                st.dataframe(signals[display_cols], use_container_width=True)

        # ── TAB 2: CONFLUENCE ──
        with tab2:
            st.subheader("Confluence Scored Setups")
            st.caption("Signals ranked by how many patterns stack at the same level")

            if confluence.empty:
                st.info("No confluence signals found.")
            else:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Setups", len(confluence))
                col_b.metric("High Confluence (6+)", len(confluence[confluence["score"] >= 6]))
                col_c.metric("Strong (4-5)", len(confluence[confluence["score"].between(4, 5)]))

                st.markdown("---")

                filter_strength = st.selectbox(
                    "Filter by strength",
                    ["All", "🔥 HIGH only", "⚡ STRONG+", "👀 MODERATE+"],
                    key="conf_filter"
                )

                filtered = confluence.copy()
                if filter_strength == "🔥 HIGH only":
                    filtered = filtered[filtered["score"] >= 6]
                elif filter_strength == "⚡ STRONG+":
                    filtered = filtered[filtered["score"] >= 4]
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
            skip_keys = ["outcomes", "summary", "confluence", "fvg_behaviour", "fvg_summary"]
            for key, frame in results.items():
                if key in skip_keys:
                    continue
                if isinstance(frame, pd.DataFrame) and not frame.empty and "pattern" in frame.columns:
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
            breakdown_tabs = st.tabs(["Equal Highs/Lows", "Fair Value Gaps", "BOS / CHoCH", "Consolidation"])

            with breakdown_tabs[0]:
                st.dataframe(results["equal_highs_lows"], use_container_width=True)
            with breakdown_tabs[1]:
                st.dataframe(results["fvg"], use_container_width=True)
            with breakdown_tabs[2]:
                st.dataframe(results["bos_choch"], use_container_width=True)
            with breakdown_tabs[3]:
                st.dataframe(results["consolidation"], use_container_width=True)

        # ── TAB 4: RISK MANAGEMENT ──
        with tab4:
            st.subheader("Risk Management")
            st.caption("Average moves, MAE and R:R per pattern — use these to set your TP and SL")

            if not results["outcomes"].empty:
                risk = results["outcomes"].groupby("pattern").agg(
                    avg_up_move=("up_move", "mean"),
                    avg_down_move=("down_move", "mean"),
                    max_up_move=("up_move", "max"),
                    max_down_move=("down_move", "max"),
                    avg_mae=("mae", "mean"),
                    max_mae=("mae", "max"),
                ).reset_index()

                risk["avg_rr_ratio"] = (risk["avg_up_move"] / risk["avg_down_move"]).round(2)
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

        # ── TAB 5: FVG DEEP ANALYSIS ──
        with tab5:
            st.subheader("Bullish FVG — Deep Behaviour Analysis")
            st.caption("Every Bullish FVG in the dataset analysed for failure, drawdown and TP range")

            fvg_summary = results.get("fvg_summary", {})
            fvg_behaviour = results.get("fvg_behaviour", pd.DataFrame())

            if not fvg_summary:
                st.info("No FVG behaviour data available.")
            else:
                st.markdown("### Overview")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Bullish FVGs", fvg_summary["total_fvgs"])
                m2.metric("Failures", f"{fvg_summary['failure_count']} ({fvg_summary['failure_rate']}%)")
                m3.metric("Avg Drawdown", fvg_summary["avg_drawdown"])
                m4.metric("Max Drawdown", fvg_summary["max_drawdown"])

                st.markdown("---")
                st.markdown("### Failure Analysis")
                st.caption("A failure = price came into the FVG, traded below gap bottom and hit SL")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background-color:#1e1e2e; padding:15px; border-radius:8px; border-left:5px solid #ef553b">
                        <h4 style="color:#ef553b">FVG Failure Rate</h4>
                        <h2 style="color:white">{fvg_summary['failure_rate']}%</h2>
                        <p style="color:#aaa">{fvg_summary['failure_count']} failures out of {fvg_summary['total_fvgs']} total FVGs</p>
                        <p style="color:#aaa">Average failure occurred on candle <b style="color:white">{fvg_summary['avg_failure_candle']}</b> after entry</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background-color:#1e1e2e; padding:15px; border-radius:8px; border-left:5px solid #00cc96">
                        <h4 style="color:#00cc96">FVG Success Rate</h4>
                        <h2 style="color:white">{round(100 - fvg_summary['failure_rate'], 2)}%</h2>
                        <p style="color:#aaa">{fvg_summary['total_fvgs'] - fvg_summary['failure_count']} successful FVGs</p>
                        <p style="color:#aaa">Price held above gap bottom and moved up</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### Drawdown Analysis")
                st.caption("How far did price drop from entry before recovering — on ALL FVGs")

                col3, col4, col5 = st.columns(3)
                col3.metric("Avg Drawdown", fvg_summary["avg_drawdown"])
                col4.metric("Max Drawdown Ever", fvg_summary["max_drawdown"])
                col5.metric("FVGs with above avg drawdown", fvg_summary["drawdown_over_avg"])

                if not fvg_behaviour.empty:
                    fig_dd = px.histogram(
                        fvg_behaviour,
                        x="drawdown",
                        nbins=30,
                        title="Drawdown Distribution — All Bullish FVGs",
                        color_discrete_sequence=["#ef553b"]
                    )
                    fig_dd.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_dd, use_container_width=True, key="fig_dd")

                st.markdown("---")
                st.markdown("### TP Range Analysis")
                st.caption("How far did price move up after FVG formed and for how many candles")

                col6, col7, col8 = st.columns(3)
                col6.metric("Avg Up Move", fvg_summary["avg_up_move"])
                col7.metric("Max Up Move Ever", fvg_summary["max_up_move"])
                col8.metric("Avg Candles Up Before Retrace", fvg_summary["avg_candles_up"])

                buckets = fvg_summary["tp_buckets"]
                bucket_df = pd.DataFrame({
                    "Range": list(buckets.keys()),
                    "Count": list(buckets.values())
                })

                fig_tp = px.bar(
                    bucket_df,
                    x="Range",
                    y="Count",
                    color="Count",
                    text="Count",
                    title="How Far Did Price Move After Bullish FVG",
                    color_continuous_scale="Greens"
                )
                fig_tp.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-20
                )
                st.plotly_chart(fig_tp, use_container_width=True, key="fig_tp")

                st.markdown(f"""
                <div style="background-color:#1e1e2e; padding:15px; border-radius:8px; border-left:5px solid #f0c040; margin-top:10px">
                    <h4 style="color:#f0c040">Optimal TP Recommendation</h4>
                    <p style="color:#aaa">Based on <b style="color:white">{fvg_summary['total_fvgs']}</b> Bullish FVGs analysed:</p>
                    <p style="color:white">Average move after FVG: <b style="color:#00cc96">{fvg_summary['avg_up_move']}</b> points</p>
                    <p style="color:white">Maximum move recorded: <b style="color:#00cc96">{fvg_summary['max_up_move']}</b> points</p>
                    <p style="color:white">Price stayed up for average of <b style="color:#00cc96">{fvg_summary['avg_candles_up']}</b> candles before retracing</p>
                    <p style="color:white">Recommended TP: <b style="color:#00cc96">{fvg_summary['avg_up_move']}</b> | Conservative TP: <b style="color:#00cc96">{round(fvg_summary['avg_up_move'] * 0.7, 2)}</b></p>
                    <p style="color:white">Recommended SL buffer beyond gap bottom: <b style="color:#ef553b">{fvg_summary['avg_drawdown']}</b> points</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                with st.expander("View all FVG instances"):
                    st.dataframe(fvg_behaviour, use_container_width=True)

        # ── TAB 6: RAW DATA ──
        with tab6:
            st.subheader("Raw Candle Data")
            st.caption(f"{len(df)} candles — {symbol} {timeframe}")
            st.dataframe(df, use_container_width=True)