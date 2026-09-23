"""Plotly chart builders for the dashboard."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_chart(df, symbol: str, indicators: dict = None):
    """Main candlestick chart with overlays (MA, EMA, Bollinger, VWAP, Pineify, Smart VWAP)."""
    if df.empty:
        return go.Figure()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if "sma_short" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_short"], name="Pineify Trend (SMA 20)", line=dict(color="#2962ff", width=1)), row=1, col=1)
    if "sma_long" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_long"], name="Pineify Base (SMA 50)", line=dict(color="#ff6d00", width=1)), row=1, col=1)
    if "ema_short" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_short"], name="Pineify Momentum (EMA 9)", line=dict(color="#00bfa5", width=1, dash="dot")), row=1, col=1)
    if "smart_vwap" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["smart_vwap"], name="Smart VWAP", line=dict(color="#7e57c2", width=2)), row=1, col=1)
    if "vwap" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], name="VWAP", line=dict(color="#b39ddb", width=1, dash="dash")), row=1, col=1)
    if "bb_upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="#b0bec5", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="#b0bec5", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], name="BB Mid", line=dict(color="#90a4ae", width=1)), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors), row=2, col=1)

    if "rsi" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#f9a825", width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Price, Pineify Signals & Smart VWAP",
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig


def equity_chart(equity_curve):
    """Equity curve + close price comparison for backtest."""
    if equity_curve is None or equity_curve.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["equity"], name="Strategy Equity", line=dict(color="#26a69a", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["close"], name="Close Price", line=dict(color="#7e57c2", width=1)), secondary_y=True)
    fig.update_layout(title="Backtest Equity Curve", template="plotly_dark", height=400,
                      margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text="Equity ($)", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True)
    return fig


def gauge_chart(value: float, title: str):
    """A simple indicator gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#2962ff"},
               "steps": [
                   {"range": [0, 38], "color": "#ef5350"},
                   {"range": [38, 62], "color": "#ffb300"},
                   {"range": [62, 100], "color": "#26a69a"},
               ]},
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
    return fig