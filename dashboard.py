"""
dashboard.py
------------
Live Streamlit dashboard — Analysis + Forecasting tabs.
 
Usage:
    streamlit run dashboard.py
"""
 
import os
import sys
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import asdict
 
# ── SET YOUR GROQ KEY HERE ────────────────────────────────────────────────────
GROQ_API_KEY = "your_groq_key_here"
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
DB_PATH = "data/nifty50.db"
 
st.set_page_config(
    page_title="AI Hedge Fund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
.approved { color: #00c853; font-weight: bold; }
.blocked  { color: #ff1744; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
 
 
# ── HELPERS ───────────────────────────────────────────────────────────────────
 
@st.cache_data(ttl=300)
def get_symbols():
    conn    = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT Symbol FROM prices ORDER BY Symbol"
    ).fetchall()]
    conn.close()
    return symbols
 
 
@st.cache_data(ttl=60)
def get_price_history(symbol, days=180):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Open, High, Low, Close, Volume, Daily_Return
        FROM prices WHERE Symbol='{symbol}'
        ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    return df.sort_values("Date").reset_index(drop=True)
 
 
def get_equity_curve():
    path = "data/equity_curve.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)
 
 
def get_trade_log():
    path = "data/trade_log.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)
 
 
# ── CHARTS ────────────────────────────────────────────────────────────────────
 
def candlestick_chart(df, symbol):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00c853",
        decreasing_line_color="#ff1744",
        name=symbol,
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"],
        line=dict(color="#2196f3", width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"],
        line=dict(color="#ff9800", width=1.5), name="MA50"))
    fig.update_layout(
        title=f"{symbol} — Price Chart",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=420,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig
 
 
def forecast_chart(fc):
    """Build the GBM forecast chart with confidence bands."""
    fig = go.Figure()
 
    all_dates  = fc["hist_dates"] + fc["future_dates"]
    hist_n     = len(fc["hist_dates"])
    future_n   = len(fc["future_dates"])
 
    # Historical price
    fig.add_trace(go.Scatter(
        x=fc["hist_dates"], y=fc["hist_close"],
        line=dict(color="#aaaaaa", width=1.5),
        name="Historical Price",
    ))
 
    # Kalman smoothed historical
    fig.add_trace(go.Scatter(
        x=fc["hist_dates"], y=fc["smoothed_hist"],
        line=dict(color="#2196f3", width=2, dash="dot"),
        name="Kalman Smoothed",
    ))
 
    # Confidence bands — build from future dates only
    fd = fc["future_dates"]
    p5  = fc["p5"][1:]   # remove index 0 (current price)
    p25 = fc["p25"][1:]
    p50 = fc["p50"][1:]
    p75 = fc["p75"][1:]
    p95 = fc["p95"][1:]
 
    # 90% confidence band (5-95)
    fig.add_trace(go.Scatter(
        x=fd + fd[::-1],
        y=p95 + p5[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence",
        showlegend=True,
    ))
 
    # 50% confidence band (25-75)
    fig.add_trace(go.Scatter(
        x=fd + fd[::-1],
        y=p75 + p25[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="50% Confidence",
        showlegend=True,
    ))
 
    # Median forecast line
    fig.add_trace(go.Scatter(
        x=fd, y=p50,
        line=dict(color="#00c853", width=2.5, dash="dash"),
        name="Base Case (Median)",
    ))
 
    # Bull case
    fig.add_trace(go.Scatter(
        x=fd, y=p75,
        line=dict(color="#69f0ae", width=1.5, dash="dot"),
        name=f"Bull Case (+{fc['bull_ret']:+.1f}%)",
    ))
 
    # Bear case
    fig.add_trace(go.Scatter(
        x=fd, y=p25,
        line=dict(color="#ff5252", width=1.5, dash="dot"),
        name=f"Bear Case ({fc['bear_ret']:+.1f}%)",
    ))
 
    # Support and resistance lines
    fig.add_hline(
        y=fc["resistance"], line_dash="dash",
        line_color="#ff9800", opacity=0.7,
        annotation_text=f"Resistance ₹{fc['resistance']}",
        annotation_position="right",
    )
    fig.add_hline(
        y=fc["support"], line_dash="dash",
        line_color="#2196f3", opacity=0.7,
        annotation_text=f"Support ₹{fc['support']}",
        annotation_position="right",
    )
 
    # Current price line
    fig.add_hline(
        y=fc["current_price"], line_dash="solid",
        line_color="#ffffff", opacity=0.3,
        annotation_text=f"Now ₹{fc['current_price']}",
        annotation_position="left",
    )
 
    # Vertical line separating history from forecast
    fig.add_shape(
    type="line",
    x0=fc["hist_dates"][-1], x1=fc["hist_dates"][-1],
    y0=0, y1=1, yref="paper",
    line=dict(color="#555555", dash="dash", width=1),
)
    fig.add_annotation(
    x=fc["hist_dates"][-1], y=1, yref="paper",
    text="Today", showarrow=False,
    font=dict(color="#aaaaaa", size=11),
    xanchor="left",
)
 
    fig.update_layout(
        title=f"{fc['symbol']} — {fc['n_sims']:,}-Simulation GBM Forecast ({fc['horizon_days']}-day)",
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=520,
        margin=dict(t=50, b=20, l=20, r=80),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e1e2e", showgrid=True)
    fig.update_yaxes(gridcolor="#1e1e2e", showgrid=True, tickprefix="₹")
    return fig
 
 
def equity_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["portfolio_value"],
        fill="tozeroy", line=dict(color="#00c853", width=2),
        fillcolor="rgba(0,200,83,0.1)", name="Portfolio",
    ))
    fig.update_layout(
        title="Portfolio Equity Curve",
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig
 
 
def signal_bar(score, label, note=""):
    color = "#00c853" if score >= 0 else "#ff1744"
    width = f"{abs(score) * 100:.0f}%"
    left  = "50%" if score >= 0 else f"{(0.5 + score/2)*100:.0f}%"
    st.markdown(f"""
    <div style='margin-bottom:12px'>
        <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
            <span style='font-size:13px;color:#aaa'>{label}</span>
            <span style='font-size:13px;font-weight:bold;color:{color}'>{score:+.4f}</span>
        </div>
        <div style='background:#2a2a3e;border-radius:4px;height:8px;position:relative'>
            <div style='position:absolute;left:50%;width:1px;height:8px;background:#555'></div>
            <div style='position:absolute;left:{left};width:{width};height:8px;
                        background:{color};border-radius:4px'></div>
        </div>
        <div style='font-size:11px;color:#666;margin-top:2px'>{note}</div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ── LLM ───────────────────────────────────────────────────────────────────────
 
def call_groq(prompt):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quant analyst. Respond ONLY in valid JSON with keys: final_score (float -1 to 1), recommendation (STRONG BUY/BUY/HOLD/SELL/STRONG SELL), thesis (2-3 sentences), risks (1-2 sentences), confidence (HIGH/MEDIUM/LOW)."
                },
                {"role": "user", "content": prompt}
            ]
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise Exception(f"Groq error {resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())
 
 
# ── ANALYSIS PIPELINE ─────────────────────────────────────────────────────────
 
def run_analysis(symbol):
    try:
        from agents.regime_agent                  import get_regime
        from agents.kalman_momentum_agent         import get_latest_signal as kalman_sig
        from agents.advanced_mean_reversion_agent import get_latest_signal as adv_mr_sig
        from agents.ml_agent                      import get_latest_signal as ml_sig
        from agents.gbm_monte_carlo_agent         import get_latest_signal as mc_sig
        from risk.risk_engine                     import evaluate_trade, Portfolio, RiskConfig
 
        with st.spinner("Detecting market regime (HMM)..."):
            try:
                regime  = get_regime()
                weights = regime["agent_weights"]
            except Exception as e:
                regime  = {
                    "regime": 0, "regime_name": "low volatility bull",
                    "probabilities": [1.0, 0.0, 0.0],
                    "description": "Default bull regime",
                    "agent_weights": {"momentum": 0.35, "mean_reversion": 0.30, "ml_classifier": 0.35}
                }
                weights = regime["agent_weights"]
 
        signals = {}
        with st.spinner("Running Kalman momentum agent..."):
            try:    signals["kalman"] = kalman_sig(symbol)
            except: signals["kalman"] = {"score": 0.0, "interpretation": "failed"}
 
        with st.spinner("Running advanced mean reversion (~60s)..."):
            try:    signals["adv_mr"] = adv_mr_sig(symbol)
            except: signals["adv_mr"] = {"score": 0.0, "interpretation": "failed", "hurst": 0.5}
 
        with st.spinner("Training ML classifier..."):
            try:    signals["ml"] = ml_sig(symbol)
            except: signals["ml"] = {"score": 0.0, "interpretation": "failed"}
 
        with st.spinner("Running GBM Monte Carlo..."):
            try:    signals["monte_carlo"] = mc_sig(symbol)
            except: signals["monte_carlo"] = {"score": 0.0, "interpretation": "failed",
                                               "prob_gain": 0.5, "expected_ret": 0.0, "cvar_5pct": 0.0}
 
        mc_weight    = 0.20
        regime_scale = 1.0 - mc_weight
        combined = float(np.clip(
            weights["momentum"]       * regime_scale * signals["kalman"]["score"]  +
            weights["mean_reversion"] * regime_scale * signals["adv_mr"]["score"]  +
            weights["ml_classifier"]  * regime_scale * signals["ml"]["score"]      +
            mc_weight * signals["monte_carlo"]["score"],
            -1, 1
        ))
        signals["combined"] = round(combined, 4)
 
        with st.spinner("Calling Groq LLM..."):
            try:
                mc     = signals["monte_carlo"]
                mr     = signals["adv_mr"]
                prompt = f"""Analyze {symbol}.
REGIME: {regime['regime_name']} — {regime['description']}
Kalman Momentum: {signals['kalman']['score']:+.4f} | {signals['kalman']['interpretation']}
Adv Mean Rev: {mr['score']:+.4f} | H={mr.get('hurst','?')} | {mr['interpretation']}
ML Classifier: {signals['ml']['score']:+.4f} | {signals['ml']['interpretation']}
GBM Monte Carlo: {mc['score']:+.4f} | P(gain)={mc.get('prob_gain',0):.1%} | CVaR={mc.get('cvar_5pct',0):.2f}%
Combined: {combined:+.4f}"""
                llm = call_groq(prompt)
            except Exception as e:
                rec = "BUY" if combined > 0.1 else ("SELL" if combined < -0.1 else "HOLD")
                mc  = signals["monte_carlo"]
                mr  = signals["adv_mr"]
                llm = {
                    "final_score": combined, "recommendation": rec,
                    "thesis": f"Combined score {combined:+.4f}: Kalman={signals['kalman']['score']:+.3f}, MeanRev={mr['score']:+.3f}, ML={signals['ml']['score']:+.3f}, MC P(gain)={mc.get('prob_gain',0):.1%}.",
                    "risks":  f"Hurst={mr.get('hurst','?')} ({'trending' if mr.get('hurst',0.5)>0.6 else 'mean-reverting'}). CVaR={mc.get('cvar_5pct',0):.2f}%.",
                    "confidence": "MEDIUM",
                }
 
        conn  = sqlite3.connect(DB_PATH)
        price = float(conn.execute(
            "SELECT Close FROM prices WHERE Symbol=? ORDER BY Date DESC LIMIT 1", (symbol,)
        ).fetchone()[0])
        conn.close()
 
        portfolio = Portfolio(cash=100_000)
        config    = RiskConfig(total_capital=100_000)
        trade = evaluate_trade(
            symbol=symbol, signal_score=llm["final_score"],
            current_price=price, portfolio=portfolio, config=config,
            regime=regime["regime"], confidence=llm.get("confidence", "MEDIUM"),
            hurst=float(signals["adv_mr"].get("hurst", 0.5)),
            cvar_mc=float(signals["monte_carlo"].get("cvar_5pct", 0.0)),
        )
 
        return {"regime": regime, "signals": signals, "llm": llm,
                "trade": asdict(trade), "price": price}
 
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    st.sidebar.title("📈 AI Hedge Fund")
    st.sidebar.markdown("---")
 
    symbols        = get_symbols()
    selected_stock = st.sidebar.selectbox("Select Stock", symbols)
 
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Actions**")
    analysis_btn  = st.sidebar.button("🔍 Run Full Analysis",  type="primary",   use_container_width=True)
    forecast_btn  = st.sidebar.button("🔮 Run Forecasting",    type="secondary", use_container_width=True)
 
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline**")
    st.sidebar.markdown("HMM → Kalman → ADF → ML → GBM → LLM → Risk")
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
 
    st.title("AI Hedge Fund Dashboard")
    st.caption("Nifty 50 · Multi-Agent System · GBM Forecasting · 10,000 Simulations")
    st.markdown("---")
 
    # Price chart always visible
    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        price_df = get_price_history(selected_stock)
        if not price_df.empty:
            st.plotly_chart(candlestick_chart(price_df, selected_stock), use_container_width=True)
 
    with col_info:
        if not price_df.empty:
            latest = price_df.iloc[-1]
            prev   = price_df.iloc[-2] if len(price_df) > 1 else latest
            change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
            st.metric(f"{selected_stock}", f"₹{latest['Close']:.2f}", f"{change:+.2f}%")
            st.metric("52W High", f"₹{price_df['High'].max():.2f}")
            st.metric("52W Low",  f"₹{price_df['Low'].min():.2f}")
            st.metric("Avg Volume", f"{price_df['Volume'].mean()/1e6:.1f}M")
 
    # ── FORECASTING TAB ───────────────────────────────────────────────────────
    if forecast_btn:
        st.markdown("---")
        st.markdown("## 🔮 30-Day Price Forecast")
        st.caption(f"GBM Monte Carlo · {1000:,} simulations · Kalman smoothed trend · Support/Resistance levels")
 
        with st.spinner(f"Running 1,000 GBM simulations for {selected_stock}..."):
            from forecaster import run_forecast
            fc = run_forecast(selected_stock)
 
        if fc:
            # Forecast chart
            st.plotly_chart(forecast_chart(fc), use_container_width=True)
 
            st.markdown("---")
 
            # Trend and scenarios
            t1, t2, t3, t4 = st.columns(4)
            t1.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'>
                <div style='font-size:0.85rem;color:#aaa'>Current Trend</div>
                <div style='font-size:1.1rem;font-weight:bold;color:{fc["trend_color"]}'>{fc["trend"]}</div>
                <div style='font-size:0.75rem;color:#666'>strength={fc["trend_strength"]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
 
            t2.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'>
                <div style='font-size:0.85rem;color:#aaa'>🐂 Bull Case (30d)</div>
                <div style='font-size:1.1rem;font-weight:bold;color:#00c853'>₹{fc["bull_target"]}</div>
                <div style='font-size:0.85rem;color:#00c853'>{fc["bull_ret"]:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
 
            t3.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'>
                <div style='font-size:0.85rem;color:#aaa'>📊 Base Case (30d)</div>
                <div style='font-size:1.1rem;font-weight:bold;color:#ffa000'>₹{fc["base_target"]}</div>
                <div style='font-size:0.85rem;color:#ffa000'>{fc["base_ret"]:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
 
            t4.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'>
                <div style='font-size:0.85rem;color:#aaa'>🐻 Bear Case (30d)</div>
                <div style='font-size:1.1rem;font-weight:bold;color:#ff1744'>₹{fc["bear_target"]}</div>
                <div style='font-size:0.85rem;color:#ff1744'>{fc["bear_ret"]:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            # Probabilities and levels
            p1, p2, p3 = st.columns(3)
 
            with p1:
                st.markdown("**Probability Analysis**")
                st.metric("P(Price Goes Up)",     f"{fc['prob_up']}%")
                st.metric("P(+10% in 30 days)",   f"{fc['prob_10up']}%")
                st.metric("P(-10% in 30 days)",   f"{fc['prob_10dn']}%")
 
            with p2:
                st.markdown("**Expected Returns**")
                st.metric("Expected Return (30d)", f"{fc['expected_ret']:+.2f}%")
                st.metric("CVaR 5% (worst case)",  f"{fc['cvar_5pct']:.2f}%")
                st.metric("Daily Volatility σ",    f"{fc['sigma_daily']:.2f}%")
 
            with p3:
                st.markdown("**Key Levels**")
                st.metric("Resistance", f"₹{fc['resistance']}")
                st.metric("Pivot",      f"₹{fc['pivot']}")
                st.metric("Support",    f"₹{fc['support']}")
 
            st.markdown("---")
            st.caption(
                f"Model: Geometric Brownian Motion | "
                f"Drift μ={fc['mu_daily']:+.4f}%/day | "
                f"Volatility σ={fc['sigma_daily']:.4f}%/day | "
                f"{fc['n_sims']:,} simulations | "
                f"Parameters estimated from last 60 trading days"
            )
 
    # ── ANALYSIS TAB ──────────────────────────────────────────────────────────
    if analysis_btn:
        st.markdown("---")
        st.markdown("## 🔍 Full Analysis")
        result = run_analysis(selected_stock)
 
        if result:
            regime = result["regime"]
            icons  = {"low volatility bull": "🟢", "high volatility bull": "🟡", "bear / crisis": "🔴"}
            icon   = icons.get(regime["regime_name"], "⚪")
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Regime", f"{icon} {regime['regime_name'].title()}")
            c2.metric("Bull Probability", f"{regime['probabilities'][0]:.0%}")
            c3.metric("Bear Probability", f"{regime['probabilities'][2]:.0%}")
            st.caption(regime["description"])
            st.markdown("---")
 
            sig_col, dec_col = st.columns(2)
 
            with sig_col:
                st.markdown("### Agent Signals")
                s = result["signals"]
                signal_bar(s["kalman"]["score"],      "Kalman Momentum",   s["kalman"]["interpretation"])
                signal_bar(s["adv_mr"]["score"],      "Advanced Mean Rev", s["adv_mr"]["interpretation"])
                signal_bar(s["ml"]["score"],          "ML Classifier",     s["ml"]["interpretation"])
                signal_bar(s["monte_carlo"]["score"], "GBM Monte Carlo",   s["monte_carlo"]["interpretation"])
                st.markdown("---")
                signal_bar(s["combined"],             "Combined Score",    "")
 
                mc = s["monte_carlo"]
                st.markdown("**Monte Carlo (10,000 sims)**")
                m1, m2, m3 = st.columns(3)
                m1.metric("P(Gain)",   f"{mc.get('prob_gain',0):.1%}")
                m2.metric("E[Return]", f"{mc.get('expected_ret',0):+.2f}%")
                m3.metric("CVaR 5%",   f"{mc.get('cvar_5pct',0):.2f}%")
 
            with dec_col:
                st.markdown("### Trade Decision")
                llm   = result["llm"]
                trade = result["trade"]
 
                rec_colors = {
                    "STRONG BUY": "#00c853", "BUY": "#69f0ae",
                    "HOLD": "#ffa000",
                    "SELL": "#ff5252", "STRONG SELL": "#ff1744",
                }
                color = rec_colors.get(llm["recommendation"], "#aaa")
 
                st.markdown(f"""
                <div style='background:#1e1e2e;border-radius:12px;padding:1rem;
                            margin-bottom:1rem;border-left:4px solid {color}'>
                    <div style='font-size:1.4rem;font-weight:bold;color:{color}'>
                        {llm['recommendation']}
                    </div>
                    <div style='font-size:0.85rem;color:#aaa;margin-top:4px'>
                        Score: {llm['final_score']:+.4f} | Confidence: {llm.get('confidence','?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
                st.markdown("**Trade Thesis**")
                st.info(llm.get("thesis", "N/A"))
 
                st.markdown("**Key Risks**")
                st.warning(llm.get("risks", "N/A"))
 
                st.markdown("**Risk Engine Decision**")
                if trade["approved"]:
                    st.success(
                        f"✓ APPROVED — {trade['action']} {trade['shares']} shares "
                        f"@ ₹{result['price']:.2f}\n\n"
                        f"Capital: ₹{trade['capital_allocated']:,.0f} "
                        f"({trade['position_size_pct']:.1%} of portfolio)"
                    )
                else:
                    st.error(f"✗ BLOCKED — {trade['rejection_reason']}")
 
                if trade["risk_notes"]:
                    with st.expander("Risk Engine Notes"):
                        for note in trade["risk_notes"]:
                            st.caption(f"• {note}")
 
                H  = result["signals"]["adv_mr"].get("hurst", 0.5)
                hc = "#00c853" if H < 0.5 else ("#ffa000" if H < 0.6 else "#ff1744")
                lb = "mean-reverting" if H < 0.5 else ("random walk" if H < 0.6 else "trending")
                st.markdown("**Hurst Exponent**")
                st.markdown(f"""
                <div style='background:#1e1e2e;border-radius:8px;padding:0.75rem'>
                    <span style='font-size:1.2rem;font-weight:bold;color:{hc}'>{H:.3f}</span>
                    <span style='color:#aaa;margin-left:8px'>{lb}</span>
                </div>
                """, unsafe_allow_html=True)
 
    # Bottom
    st.markdown("---")
    eq_col, tl_col = st.columns(2)
 
    with eq_col:
        st.markdown("### Portfolio Equity Curve")
        eq_df = get_equity_curve()
        if eq_df is not None:
            st.plotly_chart(equity_chart(eq_df), use_container_width=True)
            final = eq_df["portfolio_value"].iloc[-1]
            ret   = (final - 100_000) / 100_000 * 100
            c1, c2 = st.columns(2)
            c1.metric("Portfolio Value", f"₹{final:,.0f}")
            c2.metric("Total Return",    f"{ret:+.2f}%")
        else:
            st.info("Run `python3.14 backtester_v2.py` first.")
 
    with tl_col:
        st.markdown("### Recent Trades")
        tl_df = get_trade_log()
        if tl_df is not None:
            display = tl_df.tail(20).copy()
            if "pnl" in display.columns:
                display["pnl"] = display["pnl"].fillna(0)
            st.dataframe(display, use_container_width=True, height=300)
        else:
            st.info("Run `python3.14 backtester_v2.py` first.")
 
 
if __name__ == "__main__":
    main()
 