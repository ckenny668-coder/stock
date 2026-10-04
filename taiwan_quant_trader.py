# -*- coding: utf-8 -*-
"""
台股量化選股／買賣訊號系統（修正版）
============================================================
功能：
1. 技術分析：K線、MA、MACD、KD、RSI、BIAS、布林通道
2. 買賣訊號：進場、停利、停損、支撐／壓力、報酬風險比
3. 量價分析：價漲量增／縮、價跌量增／縮
4. 選股策略：綜合、短線、波段、突破、回檔、量價突破
5. 多檔掃描：輸入股票清單，一次掃描最新訊號
6. 籌碼分析：三大法人買賣超（FinMind）
7. 基本面：月營收與年增率（FinMind）
8. 回測：隔日開盤成交、跳空停損、整張／零股、勝率、最大回撤、Sharpe
9. 風險管理：部位大小、最大虧損、停損、停利、報酬風險比
10. Plotly 互動圖表

安裝：
    pip install pandas numpy plotly streamlit FinMind twstock

啟動：
    streamlit run taiwan_quant_trader.py

注意：
- 本程式為研究／教育用途，不構成投資建議。
- 不使用 yfinance。
- FinMind 欄位可能隨 API 版本調整，籌碼／營收讀不到時會略過，不影響技術分析。
"""

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

try:
    from FinMind.data import DataLoader
    FINMIND_OK = True
except Exception:
    FINMIND_OK = False

try:
    import twstock
    TWSTOCK_OK = True
except Exception:
    TWSTOCK_OK = False


# =========================================================
# Streamlit 版本相容
# =========================================================
def show_df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


UI = {"mobile": True}  # 由側邊欄「手機版面」勾選決定


def show_chart(fig, **kwargs) -> None:
    if UI["mobile"]:
        h = fig.layout.height or 450
        fig.update_layout(
            height=min(h, 480 if h > 600 else 340),
            margin=dict(l=8, r=8, t=40, b=8),
            legend=dict(orientation="h", y=-0.15),
        )
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def metric_grid(items, per_row: int = 4) -> None:
    """items: [(名稱, 已格式化文字)]。手機版用緊湊表格，電腦版用 metric 卡片。"""
    if UI["mobile"]:
        show_df(pd.DataFrame({"指標": [n for n, _ in items], "數值": [v for _, v in items]}))
        return
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        for c, (name, value) in zip(st.columns(len(chunk)), chunk):
            c.metric(name, value)


# =========================================================
# Utility
# =========================================================
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_value(value, fmt: str) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "-"
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """將常見中英文欄位統一成 Date/Open/High/Low/Close/Volume。"""
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {
        "date": "Date", "日期": "Date",
        "open": "Open", "開盤價": "Open",
        "max": "High", "high": "High", "最高價": "High", "最高": "High",
        "min": "Low", "low": "Low", "最低價": "Low", "最低": "Low",
        "close": "Close", "收盤價": "Close",
        "Trading_Volume": "Volume", "TradingVolume": "Volume",
        "volume": "Volume", "成交量": "Volume", "成交股數": "Volume",
    }
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns}).copy()

    if "Date" not in df.columns:
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df.drop_duplicates("Date").reset_index(drop=True)

    # 缺少開高低時用收盤補，避免後續計算全 NaN
    for c in ["Open", "High", "Low"]:
        df[c] = df[c].fillna(df["Close"])

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def make_loader(token: str = ""):
    dl = DataLoader()
    if token:
        try:
            dl.login_by_token(api_token=token)
        except Exception:
            pass
    return dl


# =========================================================
# Data loading
# 注意：快取函式內失敗時「丟出例外」，這樣失敗結果不會被快取。
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def _finmind_daily(stock_id: str, start: str, end: str, token: str) -> pd.DataFrame:
    if not FINMIND_OK:
        raise RuntimeError("FinMind 未安裝")
    dl = make_loader(token)
    df = dl.taiwan_stock_daily(stock_id=str(stock_id), start_date=start, end_date=end)
    df = normalize_columns(df)
    if df.empty:
        raise ValueError("FinMind 沒有回傳資料")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def _twstock_daily(stock_id: str, start: str, end: str) -> pd.DataFrame:
    if not TWSTOCK_OK:
        raise RuntimeError("twstock 未安裝")
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    stock = twstock.Stock(str(stock_id))
    raw = stock.fetch_from(s.year, s.month)

    rows = []
    for x in raw:
        # twstock Data 欄位：date, capacity, turnover, open, high, low, close, change, transaction
        rows.append({
            "Date": pd.to_datetime(x.date),
            "Open": safe_float(x.open),
            "High": safe_float(x.high),
            "Low": safe_float(x.low),
            "Close": safe_float(x.close),
            "Volume": safe_float(x.capacity),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("twstock 沒有回傳資料")
    df = df[(df["Date"] >= s) & (df["Date"] <= e)]
    df = normalize_columns(df)
    if df.empty:
        raise ValueError("twstock 日期區間內沒有資料")
    return df


def load_daily(stock_id: str, start: str, end: str, source: str, token: str = "") -> pd.DataFrame:
    order = ["FinMind", "twstock"] if source == "FinMind" else ["twstock", "FinMind"]
    for src in order:
        try:
            if src == "FinMind":
                return _finmind_daily(stock_id, start, end, token)
            return _twstock_daily(stock_id, start, end)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def _finmind_chip(stock_id: str, start: str, end: str, token: str) -> pd.DataFrame:
    if not FINMIND_OK:
        raise RuntimeError("FinMind 未安裝")
    dl = make_loader(token)
    df = dl.taiwan_stock_institutional_investors(
        stock_id=str(stock_id), start_date=start, end_date=end
    )
    if df is None or df.empty:
        raise ValueError("無法人資料")

    df = df.copy()
    if {"buy", "sell"}.issubset(df.columns):
        df["NetBuySell"] = (
            pd.to_numeric(df["buy"], errors="coerce")
            - pd.to_numeric(df["sell"], errors="coerce")
        )
    else:
        raise ValueError("法人資料缺少 buy/sell 欄位")

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    if "name" in df.columns:
        df["Institution"] = df["name"]
    elif "institutional_investors" in df.columns:
        df["Institution"] = df["institutional_investors"]
    else:
        df["Institution"] = "法人"

    return df[["Date", "Institution", "NetBuySell"]].dropna(subset=["Date"])


@st.cache_data(ttl=3600, show_spinner=False)
def _finmind_revenue(stock_id: str, start: str, end: str, token: str) -> pd.DataFrame:
    if not FINMIND_OK:
        raise RuntimeError("FinMind 未安裝")
    dl = make_loader(token)
    # 多抓一年，才算得出區間起點的年增率
    start_ext = str(pd.to_datetime(start) - pd.Timedelta(days=400))[:10]
    df = dl.taiwan_stock_monthly_revenue(
        stock_id=str(stock_id), start_date=start_ext, end_date=end
    )
    if df is None or df.empty or "revenue" not in df.columns:
        raise ValueError("無營收資料")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date").reset_index(drop=True)
    df["YoY"] = df["Revenue"].pct_change(12) * 100
    df = df[df["Date"] >= pd.to_datetime(start)]
    return df[["Date", "Revenue", "YoY"]]


def safe_call(fn, *args) -> pd.DataFrame:
    try:
        return fn(*args)
    except Exception:
        return pd.DataFrame()


# =========================================================
# Technical indicators
# =========================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    for n in [5, 10, 20, 60, 120, 240]:
        df[f"MA{n}"] = close.rolling(n).mean()

    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["MACD"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = (df["DIF"] - df["MACD"]) * 2

    # RSI14：無下跌時 RSI=100
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    df["RSI14"] = rsi

    # KD
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    for n in [5, 10, 20]:
        ma = close.rolling(n).mean()
        df[f"BIAS{n}"] = (close - ma) / ma * 100

    df["BB_MID"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std20
    df["BB_LOWER"] = df["BB_MID"] - 2 * std20
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan) * 100

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    df["Volume_MA5"] = volume.rolling(5).mean()
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_MA20"].replace(0, np.nan)

    df["ROC5"] = close.pct_change(5) * 100
    df["ROC20"] = close.pct_change(20) * 100

    df["Support20"] = low.rolling(20).min()
    df["Resistance20"] = high.rolling(20).max()

    # 突破／跌破：與「前一日」的 20 日高低比較（不含當天，才可能成立）
    df["Breakout20"] = close > df["Resistance20"].shift(1)
    df["Breakdown20"] = close < df["Support20"].shift(1)

    df["MA5_MA20_Golden"] = (df["MA5"] > df["MA20"]) & (df["MA5"].shift(1) <= df["MA20"].shift(1))
    df["MA5_MA20_Death"] = (df["MA5"] < df["MA20"]) & (df["MA5"].shift(1) >= df["MA20"].shift(1))

    return df


# =========================================================
# Price-volume analysis（向量化）
# =========================================================
def add_price_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PriceChangePct"] = df["Close"].pct_change() * 100
    df["VolumeChangePct"] = df["Volume"].pct_change() * 100

    p = np.sign(df["Close"].diff())
    v = np.sign(df["Volume"].diff())

    p_label = np.select([p > 0, p < 0], ["價漲", "價跌"], default="價平")
    v_label = np.select([v > 0, v < 0], ["量增", "量縮"], default="量平")

    label = pd.Series(p_label, index=df.index) + pd.Series(v_label, index=df.index)
    label.iloc[0] = "資料不足"
    df["PriceVolume"] = label
    return df


# =========================================================
# Signal engine（每個因子只計一次）
# =========================================================
def generate_signals(
    df: pd.DataFrame,
    strategy: str = "綜合策略",
    rsi_buy: float = 55,
    rsi_sell: float = 70,
    volume_multiple: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()

    def b(s):
        return s.fillna(False).astype(bool)

    trend_up = b((df["Close"] > df["MA20"]) & (df["MA20"] > df["MA60"]))
    trend_down = b((df["Close"] < df["MA20"]) & (df["MA20"] < df["MA60"]))

    macd_bull = b(df["DIF"] > df["MACD"])
    macd_bear = b(df["DIF"] < df["MACD"])

    kd_bull = b((df["K"] > df["D"]) & (df["K"] < 80))
    kd_bear = b((df["K"] < df["D"]) & (df["K"] > 20))

    rsi_strong = b(df["RSI14"] > rsi_buy)
    rsi_over = b(df["RSI14"] >= rsi_sell)

    bias_buy = b(df["BIAS20"].between(-6, 2))
    bias_over = b(df["BIAS20"] > 8)

    bb_breakout = b(df["Close"] > df["BB_UPPER"])
    bb_breakdown = b(df["Close"] < df["BB_LOWER"])

    vol_break = b(df["Volume_Ratio"] >= volume_multiple)
    breakout = b(df["Breakout20"])
    breakdown = b(df["Breakdown20"])

    pullback = b(
        (df["Low"] <= df["MA20"] * 1.015)
        & (df["Close"] >= df["MA20"])
        & (df["Close"] > df["Open"])
    )

    golden = b(df["MA5_MA20_Golden"])
    death = b(df["MA5_MA20_Death"])
    above_ma20 = b(df["Close"] > df["MA20"])
    pv_up = df["PriceVolume"].eq("價漲量增")

    i = lambda s: s.astype(int)

    # ---------------- 買進分數 ----------------
    if strategy == "波段":
        buy = (i(trend_up) * 2 + i(macd_bull) + i(kd_bull) + i(rsi_strong)
               + i(bias_buy) + i(pullback) + i(golden) * 2)
    elif strategy == "突破":
        buy = (i(breakout) * 3 + i(vol_break) + i(bb_breakout) + i(trend_up) + i(macd_bull))
    elif strategy == "量價突破":
        buy = (i(breakout) * 3 + i(vol_break) * 3 + i(pv_up) + i(above_ma20))
    elif strategy == "回檔":
        buy = (i(pullback) * 3 + i(macd_bull) + i(kd_bull) + i(bias_buy) + i(trend_up))
    elif strategy == "短線":
        buy = (i(b(df["Close"] > df["MA5"])) + i(macd_bull) + i(kd_bull)
               + i(b(df["Volume_Ratio"] > 1.2)))
    else:  # 綜合策略：聯集，每個因子只計一次
        buy = (i(trend_up) * 2 + i(macd_bull) + i(kd_bull) + i(rsi_strong)
               + i(bias_buy) + i(pullback) * 2 + i(golden) * 2
               + i(breakout) * 3 + i(vol_break) * 2 + i(bb_breakout))

    # ---------------- 賣出分數 ----------------
    if strategy == "短線":
        sell = (i(b(df["Close"] < df["MA5"])) + i(macd_bear) + i(kd_bear)
                + i(b((df["Volume_Ratio"] > 1.5) & (df["Close"] < df["Close"].shift(1)))))
    else:
        sell = (i(trend_down) * 2 + i(macd_bear) + i(kd_bear) + i(rsi_over) * 2
                + i(bias_over) + i(death) * 2 + i(breakdown) * 3 + i(bb_breakdown) * 2)

    df["BuyScore"] = buy
    df["SellScore"] = sell

    threshold = 3 if strategy == "短線" else 4
    strong_th = 4 if strategy == "短線" else 6

    df["BuySignal"] = (buy >= threshold) & (buy > sell) & above_ma20
    df["SellSignal"] = (sell >= threshold) & (sell > buy)

    df["StrongBuy"] = df["BuySignal"] & ((buy >= strong_th) | breakout | golden)
    df["StrongSell"] = df["SellSignal"] & ((sell >= strong_th) | breakdown | death)

    # 訊號原因（向量化）
    conds = [
        (above_ma20, "站上MA20"),
        (b(df["MA20"] > df["MA60"]), "MA多頭"),
        (macd_bull, "MACD偏多"),
        (b(df["K"] > df["D"]), "KD偏多"),
        (vol_break, "量能放大"),
        (breakout, "突破20日壓力"),
        (pullback, "回測MA20"),
        (rsi_over, "RSI過熱"),
        (breakdown, "跌破20日支撐"),
    ]
    reasons = pd.Series("", index=df.index)
    for cond, label in conds:
        reasons = reasons + np.where(cond, label + "、", "")
    reasons = reasons.str.rstrip("、").replace("", "無明確條件")
    df["SignalReason"] = reasons

    df["Signal"] = "觀望"
    df.loc[df["BuySignal"], "Signal"] = "買進"
    df.loc[df["SellSignal"], "Signal"] = "賣出"
    df.loc[df["StrongBuy"], "Signal"] = "強力買進"
    df.loc[df["StrongSell"], "Signal"] = "強力賣出"
    return df


def analyze(df: pd.DataFrame, strategy: str, rsi_buy, rsi_sell, volume_multiple) -> pd.DataFrame:
    df = calc_indicators(df)
    df = add_price_volume(df)
    return generate_signals(df, strategy, rsi_buy, rsi_sell, volume_multiple)


# =========================================================
# Risk management
# =========================================================
def risk_plan(
    price: float,
    atr: float,
    capital: float,
    risk_pct: float,
    atr_stop_multiple: float,
    reward_risk: float,
    support: float,
    resistance: float,
    lot_size: int = 1000,
) -> Dict:
    if not np.isfinite(price) or price <= 0:
        return {}

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.03

    atr_stop = price - atr * atr_stop_multiple
    if np.isfinite(support) and support < price:
        stop = max(atr_stop, support * 0.995)
    else:
        stop = atr_stop
    stop = min(stop, price * 0.97)

    risk_per_share = max(price - stop, price * 0.005)
    max_loss = capital * risk_pct / 100.0

    by_risk = math.floor(max_loss / risk_per_share / lot_size) * lot_size
    by_cash = math.floor(capital / price / lot_size) * lot_size
    shares = max(0, min(by_risk, by_cash))

    note = ""
    if by_cash <= 0:
        note = "資金不足以買進最小單位。"
    elif by_risk <= 0:
        note = "依風險上限算出的部位小於最小單位，不建議進場（或改用零股／提高風險比例）。"
    elif by_cash < by_risk:
        note = "部位受可用資金限制，實際風險低於設定上限。"

    take_profit = price + risk_per_share * reward_risk
    if np.isfinite(resistance) and resistance > price:
        take_profit = max(take_profit, resistance)

    return {
        "Entry": price,
        "StopLoss": stop,
        "TakeProfit": take_profit,
        "RiskPerShare": risk_per_share,
        "MaxLoss": max_loss,
        "Shares": shares,
        "Lots": shares / 1000,
        "Invested": price * shares,
        "ActualRisk": risk_per_share * shares,
        "RR": (take_profit - price) / risk_per_share,
        "Note": note,
    }


# =========================================================
# Backtest：訊號隔日開盤成交；停損遇跳空以開盤價成交
# =========================================================
def backtest(
    df: pd.DataFrame,
    initial_capital: float = 1_000_000,
    fee_rate: float = 0.001425,
    tax_rate: float = 0.003,
    position_pct: float = 1.0,
    stop_loss_pct: float = 0.07,
    take_profit_pct: float = 0.15,
    lot_size: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:

    data = df.reset_index(drop=True)

    cash = initial_capital
    shares = 0
    entry_price = np.nan
    entry_date = None
    entry_cost = 0.0
    pending = None  # "buy" / "sell"，於下一根 K 棒開盤執行

    trades, equity_rows = [], []

    def close_position(date, price, reason):
        nonlocal cash, shares, entry_price, entry_date, entry_cost
        gross = shares * price
        net = gross - gross * fee_rate - gross * tax_rate
        cash += net
        profit = net - entry_cost
        trades.append({
            "EntryDate": entry_date,
            "ExitDate": date,
            "EntryPrice": entry_price,
            "ExitPrice": price,
            "Shares": shares,
            "Profit": profit,
            "ReturnPct": profit / entry_cost * 100 if entry_cost else np.nan,
            "Reason": reason,
        })
        shares = 0
        entry_price = np.nan
        entry_date = None
        entry_cost = 0.0

    for _, row in data.iterrows():
        date = row["Date"]
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]

        # 1) 執行前一日訊號（開盤價）
        if pending == "buy" and shares == 0 and o > 0:
            budget = cash * position_pct
            qty = math.floor(budget / (o * (1 + fee_rate)) / lot_size) * lot_size
            if qty > 0:
                gross = qty * o
                total = gross + gross * fee_rate
                if total <= cash:
                    cash -= total
                    shares = qty
                    entry_price = o
                    entry_date = date
                    entry_cost = total
        elif pending == "sell" and shares > 0:
            close_position(date, o, "技術賣出")
        pending = None

        # 2) 盤中停損／停利
        if shares > 0 and np.isfinite(entry_price):
            stop_price = entry_price * (1 - stop_loss_pct)
            tp_price = entry_price * (1 + take_profit_pct)

            if o <= stop_price:
                close_position(date, o, "停損（跳空）")
            elif l <= stop_price:
                close_position(date, stop_price, "停損")
            elif o >= tp_price:
                close_position(date, o, "停利（跳空）")
            elif h >= tp_price:
                close_position(date, tp_price, "停利")

        # 3) 收盤後依訊號排定下一日動作
        sig = row.get("Signal", "觀望")
        if shares == 0 and sig in ("買進", "強力買進"):
            pending = "buy"
        elif shares > 0 and sig in ("賣出", "強力賣出"):
            pending = "sell"

        equity_rows.append({
            "Date": date,
            "Cash": cash,
            "Shares": shares,
            "MarketValue": shares * c,
            "Equity": cash + shares * c,
        })

    equity = pd.DataFrame(equity_rows)

    if shares > 0 and not data.empty:
        final = data.iloc[-1]
        close_position(final["Date"], final["Close"], "回測結束平倉")
        equity.loc[equity.index[-1], "Equity"] = cash
        equity.loc[equity.index[-1], "Cash"] = cash
        equity.loc[equity.index[-1], "Shares"] = 0
        equity.loc[equity.index[-1], "MarketValue"] = 0

    trade_df = pd.DataFrame(trades)

    if equity.empty:
        return equity, trade_df, {}

    equity["Peak"] = equity["Equity"].cummax()
    equity["Drawdown"] = equity["Equity"] / equity["Peak"] - 1

    total_return = equity["Equity"].iloc[-1] / initial_capital - 1
    max_dd = equity["Drawdown"].min()

    if not trade_df.empty:
        wins = trade_df[trade_df["Profit"] > 0]
        losses = trade_df[trade_df["Profit"] <= 0]
        win_rate = len(wins) / len(trade_df)
        loss_sum = abs(losses["Profit"].sum())
        profit_factor = wins["Profit"].sum() / loss_sum if loss_sum > 0 else np.inf
        avg_trade = trade_df["ReturnPct"].mean()
    else:
        win_rate = profit_factor = avg_trade = np.nan

    daily_ret = equity["Equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        if len(daily_ret) > 1 and daily_ret.std() > 0 else np.nan
    )

    metrics = {
        "InitialCapital": initial_capital,
        "FinalEquity": equity["Equity"].iloc[-1],
        "TotalReturn": total_return * 100,
        "MaxDrawdown": max_dd * 100,
        "WinRate": win_rate * 100 if np.isfinite(win_rate) else np.nan,
        "ProfitFactor": profit_factor,
        "Sharpe": sharpe,
        "Trades": len(trade_df),
        "AverageTrade": avg_trade,
    }
    return equity, trade_df, metrics


# =========================================================
# Charts
# =========================================================
def price_chart(df: pd.DataFrame, show_volume: bool = True):
    rows = 2 if show_volume else 1
    heights = [0.78, 0.22] if show_volume else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights)

    # 台股習慣：紅漲綠跌
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="K線",
            increasing_line_color="red", increasing_fillcolor="red",
            decreasing_line_color="green", decreasing_fillcolor="green",
        ),
        row=1, col=1,
    )

    for ma in ["MA5", "MA20", "MA60", "MA120"]:
        if ma in df:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], mode="lines", name=ma), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], mode="lines",
                             name="布林上軌", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], mode="lines",
                             name="布林下軌", line=dict(dash="dot")), row=1, col=1)

    buys = df[df["BuySignal"]]
    sells = df[df["SellSignal"]]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["Date"], y=buys["Low"] * 0.98, mode="markers", name="買進訊號",
                marker=dict(symbol="triangle-up", size=11, color="red"),
                text=buys["SignalReason"],
                hovertemplate="%{x}<br>買進<br>%{text}<extra></extra>",
            ), row=1, col=1)

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["Date"], y=sells["High"] * 1.02, mode="markers", name="賣出訊號",
                marker=dict(symbol="triangle-down", size=11, color="green"),
                text=sells["SignalReason"],
                hovertemplate="%{x}<br>賣出<br>%{text}<extra></extra>",
            ), row=1, col=1)

    if show_volume:
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="成交量", opacity=0.55),
                      row=2, col=1)

    fig.update_layout(
        height=780, xaxis_rangeslider_visible=False, hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def indicator_chart(df: pd.DataFrame, indicator: str):
    fig = go.Figure()

    if indicator == "MACD":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["DIF"], name="DIF"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
        fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="柱狀體"))
    elif indicator == "KD":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["K"], name="K"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["D"], name="D"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["J"], name="J"))
        fig.add_hline(y=80, line_dash="dot")
        fig.add_hline(y=20, line_dash="dot")
    elif indicator == "RSI":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI14"))
        fig.add_hline(y=70, line_dash="dot")
        fig.add_hline(y=30, line_dash="dot")
    else:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BIAS5"], name="BIAS5"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BIAS20"], name="BIAS20"))
        fig.add_hline(y=0, line_dash="dot")

    fig.update_layout(title=indicator, height=350, hovermode="x unified",
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


# =========================================================
# Streamlit UI
# =========================================================
def main() -> None:
    st.set_page_config(page_title="台股量化選股／買賣訊號系統", page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    st.title("📈 台股量化選股／買賣訊號系統")
    st.caption("技術分析＋量價＋籌碼＋基本面＋回測＋風險管理｜不使用 yfinance")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ 系統設定")
        UI["mobile"] = st.checkbox("📱 手機版面", value=True)

        stock_id = st.text_input("股票代號", value="2330").strip()

        source = st.selectbox("資料來源", ["FinMind", "twstock"], index=0 if FINMIND_OK else 1)
        token = st.text_input("FinMind Token（選填，可提高API額度）", type="password").strip()

        today = dt.date.today()
        start_date = st.date_input("開始日期", value=today - dt.timedelta(days=900))
        end_date = st.date_input("結束日期", value=today)

        strategy = st.selectbox("選股策略", ["綜合策略", "短線", "波段", "突破", "回檔", "量價突破"])

        st.subheader("訊號參數")
        rsi_buy = st.slider("RSI強勢門檻", 45, 70, 55)
        rsi_sell = st.slider("RSI過熱門檻", 60, 90, 70)
        volume_multiple = st.slider("量能放大倍數", 1.0, 3.0, 1.5, 0.1)

        st.subheader("風險管理")
        unit = st.radio("交易單位", ["整張（1000股）", "零股（1股）"], horizontal=True)
        lot_size = 1000 if unit.startswith("整張") else 1
        capital = st.number_input("交易資金", min_value=10000.0, value=1_000_000.0, step=10000.0)
        risk_pct = st.slider("單筆最大風險 %", 0.5, 5.0, 2.0, 0.1)
        atr_stop_multiple = st.slider("ATR停損倍數", 1.0, 4.0, 2.0, 0.1)
        reward_risk = st.slider("目標報酬／風險", 1.0, 5.0, 2.0, 0.1)

        st.subheader("回測")
        initial_capital = st.number_input("回測本金", min_value=10000.0, value=1_000_000.0, step=10000.0)
        position_pct = st.slider("單次投入資金比例", 0.1, 1.0, 1.0, 0.05)
        bt_stop = st.slider("回測停損 %", 1.0, 20.0, 7.0, 0.5)
        bt_tp = st.slider("回測停利 %", 2.0, 50.0, 15.0, 1.0)

        # Streamlit 改參數會自動重算；此按鈕的用途是清除快取、強制重抓資料
        if st.button("🔄 清除快取並重新讀取"):
            st.cache_data.clear()
            st.rerun()

    # ---------------- Validate ----------------
    if not stock_id:
        st.warning("請輸入股票代號。")
        st.stop()

    if start_date >= end_date:
        st.error("開始日期必須早於結束日期。")
        st.stop()

    # ---------------- Load data ----------------
    with st.spinner("正在下載資料並計算技術指標..."):
        raw = load_daily(stock_id, str(start_date), str(end_date), source, token)

    if raw.empty:
        st.error("無法取得資料。請確認股票代號、日期區間與網路連線；FinMind 失敗時可改用 twstock。")
        st.stop()

    df = analyze(raw, strategy, rsi_buy, rsi_sell, volume_multiple)
    latest = df.iloc[-1]

    if len(df) < 70:
        st.warning(f"資料只有 {len(df)} 筆，MA60 等指標尚未穩定，訊號僅供參考。")

    # ---------------- Snapshot ----------------
    st.subheader(f"📌 {stock_id} 最新分析（{latest['Date'].date()}）")

    snapshot = [
        ("收盤價", latest["Close"], "{:.2f}"),
        ("MA20", latest["MA20"], "{:.2f}"),
        ("RSI14", latest["RSI14"], "{:.2f}"),
        ("K", latest["K"], "{:.2f}"),
        ("D", latest["D"], "{:.2f}"),
        ("BIAS20", latest["BIAS20"], "{:.2f}%"),
        ("量比", latest["Volume_Ratio"], "{:.2f}x"),
        ("訊號", latest["Signal"], "{}"),
    ]
    metric_grid([(n, fmt_value(v, f)) for n, v, f in snapshot])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 技術圖表", "💰 買賣與風控", "📊 量價分析", "🔎 歷史訊號",
        "🧮 回測", "🏦 籌碼／基本面", "🧭 多檔掃描",
    ])

    # ---------------- Tab 1 ----------------
    with tab1:
        show_chart(price_chart(df, show_volume=True))
        indicator = st.selectbox("技術指標", ["MACD", "KD", "RSI", "BIAS"], key="indicator_select")
        show_chart(indicator_chart(df, indicator))

    # ---------------- Tab 2 ----------------
    with tab2:
        price = safe_float(latest["Close"])
        atr = safe_float(latest["ATR14"])
        support = safe_float(latest["Support20"])
        resistance = safe_float(latest["Resistance20"])

        plan = risk_plan(price, atr, capital, risk_pct, atr_stop_multiple,
                         reward_risk, support, resistance, lot_size)

        if plan:
            st.subheader("🎯 交易計畫")
            metric_grid([
                ("參考進場", f"{plan['Entry']:.2f}"),
                ("停損", f"{plan['StopLoss']:.2f}"),
                ("停利", f"{plan['TakeProfit']:.2f}"),
                ("報酬／風險", f"{plan['RR']:.2f}"),
                ("建議股數", f"{plan['Shares']:,.0f}"),
                ("建議張數", f"{plan['Lots']:.3f}".rstrip("0").rstrip(".")),
                ("投入金額", f"{plan['Invested']:,.0f}"),
                ("最大風險金額", f"{plan['ActualRisk']:,.0f}"),
            ])

            if plan["Note"]:
                st.warning(plan["Note"])

            st.info(f"20日支撐：約 {support:.2f}｜20日壓力：約 {resistance:.2f}｜ATR14：約 {atr:.2f}")

            if latest["Signal"] in ("買進", "強力買進"):
                st.success(f"目前技術訊號：{latest['Signal']}｜原因：{latest['SignalReason']}")
            elif latest["Signal"] in ("賣出", "強力賣出"):
                st.error(f"目前技術訊號：{latest['Signal']}｜原因：{latest['SignalReason']}")
            else:
                st.warning(f"目前技術訊號：觀望｜條件：{latest['SignalReason']}")

        st.subheader("最近訊號")
        show_df(
            df.tail(30)[["Date", "Close", "Signal", "BuyScore", "SellScore",
                         "PriceVolume", "SignalReason"]].sort_values("Date", ascending=False)
        )

    # ---------------- Tab 3 ----------------
    with tab3:
        st.subheader("📊 量價結構")
        pv_count = df["PriceVolume"].value_counts().rename_axis("量價型態").reset_index(name="天數")

        fig_pv = go.Figure(data=[go.Bar(x=pv_count["量價型態"], y=pv_count["天數"])])
        fig_pv.update_layout(title="量價型態統計", height=350, margin=dict(l=20, r=20, t=50, b=20))
        if UI["mobile"]:
            show_chart(fig_pv)
            show_df(pv_count)
        else:
            c1, c2 = st.columns(2)
            with c1:
                show_df(pv_count)
            with c2:
                show_chart(fig_pv)

        st.subheader("最近30日量價")
        show_df(
            df.tail(30)[["Date", "Close", "Volume", "PriceChangePct", "VolumeChangePct",
                         "Volume_Ratio", "PriceVolume"]].sort_values("Date", ascending=False)
        )
        st.caption(
            "一般解讀：價漲量增代表上攻伴隨成交活躍；價漲量縮代表上漲但量能未同步放大；"
            "價跌量增代表賣壓增加；價跌量縮則需搭配支撐與趨勢判讀。"
        )

    # ---------------- Tab 4 ----------------
    with tab4:
        st.subheader("🔎 歷史訊號（單一股票）")
        st.caption("此頁為目前這檔股票在所選區間內出現過的訊號；要比較多檔請看「多檔掃描」。")

        recent = df.tail(100).copy()
        recent["CompositeScore"] = (
            recent["BuyScore"] - recent["SellScore"] + (recent["Volume_Ratio"].clip(0, 3) - 1)
        )
        selected = recent[recent["Signal"].isin(["買進", "強力買進"])].sort_values(
            ["StrongBuy", "CompositeScore"], ascending=False
        )

        if selected.empty:
            st.warning("最近100個交易日沒有符合買進條件的訊號。")
        else:
            st.success(f"最近100個交易日找到 {len(selected)} 個買進訊號（依分數排序）。")
            show_df(
                selected[["Date", "Close", "Signal", "BuyScore", "SellScore", "Volume_Ratio",
                          "RSI14", "K", "D", "BIAS20", "SignalReason"]].head(50)
            )

        st.subheader("最近歷史訊號")
        show_df(
            df[df["Signal"] != "觀望"][["Date", "Close", "Signal", "BuyScore", "SellScore",
                                      "PriceVolume", "SignalReason"]]
            .tail(50).sort_values("Date", ascending=False)
        )

    # ---------------- Tab 5 ----------------
    with tab5:
        st.subheader("🧮 策略回測")
        st.caption("訊號於收盤產生，隔日開盤價成交；停損遇跳空低開以開盤價成交；含手續費與證交稅。")

        equity, trades, metrics = backtest(
            df,
            initial_capital=initial_capital,
            position_pct=position_pct,
            stop_loss_pct=bt_stop / 100,
            take_profit_pct=bt_tp / 100,
            lot_size=lot_size,
        )

        if metrics:
            items = [
                ("最終資產", metrics["FinalEquity"], "{:,.0f}"),
                ("總報酬", metrics["TotalReturn"], "{:.2f}%"),
                ("最大回撤", metrics["MaxDrawdown"], "{:.2f}%"),
                ("勝率", metrics["WinRate"], "{:.2f}%"),
                ("Profit Factor", metrics["ProfitFactor"], "{:.2f}"),
                ("Sharpe", metrics["Sharpe"], "{:.2f}"),
                ("交易次數", metrics["Trades"], "{:.0f}"),
            ]
            metric_grid([(n, fmt_value(v, f)) for n, v, f in items], per_row=4)

            if metrics["Trades"] == 0 and lot_size == 1000:
                st.info("回測期間沒有成交。若股價偏高、本金買不起一張，可在側邊欄改成「零股」。")

        if not equity.empty:
            fig_eq = go.Figure(go.Scatter(x=equity["Date"], y=equity["Equity"], mode="lines", name="資產曲線"))
            fig_eq.update_layout(title="回測資產曲線", height=450, hovermode="x unified")
            show_chart(fig_eq)

            fig_dd = go.Figure(go.Scatter(x=equity["Date"], y=equity["Drawdown"] * 100, mode="lines", name="回撤%"))
            fig_dd.update_layout(title="最大回撤曲線", height=350, hovermode="x unified")
            show_chart(fig_dd)

        st.subheader("交易明細")
        if trades.empty:
            st.info("回測期間沒有完成交易。")
        else:
            show_df(trades.sort_values("ExitDate", ascending=False))

    # ---------------- Tab 6 ----------------
    with tab6:
        st.subheader("🏦 法人籌碼")
        chip = safe_call(_finmind_chip, stock_id, str(start_date), str(end_date), token)

        if chip.empty:
            st.warning("目前無法取得 FinMind 法人資料（可能是 API 限制或欄位變更），不影響技術分析與回測。")
        else:
            pivot = chip.groupby(["Date", "Institution"])["NetBuySell"].sum().reset_index()
            fig_chip = go.Figure()
            for inst in pivot["Institution"].dropna().unique():
                temp = pivot[pivot["Institution"] == inst]
                fig_chip.add_trace(go.Scatter(x=temp["Date"], y=temp["NetBuySell"],
                                              mode="lines", name=str(inst)))
            fig_chip.update_layout(title="法人買賣超（股）", height=450, hovermode="x unified")
            show_chart(fig_chip)
            show_df(chip.tail(50).sort_values("Date", ascending=False))

        st.subheader("💰 月營收")
        revenue = safe_call(_finmind_revenue, stock_id, str(start_date), str(end_date), token)

        if revenue.empty:
            st.warning("目前無法取得 FinMind 月營收資料。")
        else:
            fig_rev = go.Figure(go.Bar(x=revenue["Date"], y=revenue["Revenue"], name="營收"))
            fig_rev.update_layout(title="月營收", height=400, hovermode="x unified")
            show_chart(fig_rev)
            show_df(revenue.tail(24).sort_values("Date", ascending=False))

    # ---------------- Tab 7 ----------------
    with tab7:
        st.subheader("🧭 多檔掃描")
        st.caption("輸入多個股票代號（逗號、空白或換行分隔），以目前側邊欄的策略與參數掃描最新訊號。")

        default_list = "2330, 2317, 2454, 6213, 2383, 6274, 3585, 6672"
        codes_text = st.text_area("股票清單", value=default_list, height=80)

        if st.button("開始掃描"):
            codes = [c for c in pd.Series(
                codes_text.replace(",", " ").replace("，", " ").split()
            ).drop_duplicates().tolist() if c]

            if not codes:
                st.warning("請輸入至少一檔股票。")
            else:
                rows, failed = [], []
                bar = st.progress(0.0)
                for n, code in enumerate(codes, 1):
                    d = load_daily(code, str(start_date), str(end_date), source, token)
                    if d.empty or len(d) < 70:
                        failed.append(code)
                    else:
                        a = analyze(d, strategy, rsi_buy, rsi_sell, volume_multiple)
                        r = a.iloc[-1]
                        rows.append({
                            "代號": code,
                            "日期": r["Date"].date(),
                            "收盤": r["Close"],
                            "訊號": r["Signal"],
                            "買進分": r["BuyScore"],
                            "賣出分": r["SellScore"],
                            "RSI14": r["RSI14"],
                            "BIAS20": r["BIAS20"],
                            "量比": r["Volume_Ratio"],
                            "量價": r["PriceVolume"],
                            "原因": r["SignalReason"],
                        })
                    bar.progress(n / len(codes))
                bar.empty()

                if rows:
                    res = pd.DataFrame(rows).sort_values(["買進分", "賣出分"], ascending=[False, True])
                    show_df(res)
                if failed:
                    st.warning("無資料或資料不足（<70筆）：" + "、".join(failed))

    # ---------------- Footer ----------------
    st.divider()
    st.subheader("📋 最後15筆資料")
    display_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "MA60",
        "DIF", "MACD", "RSI14", "K", "D", "BIAS5", "BIAS20",
        "BB_UPPER", "BB_LOWER", "Volume_Ratio", "PriceVolume", "Signal",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    show_df(df.tail(15)[display_cols].sort_values("Date", ascending=False))

    st.caption(
        "資料來源：FinMind / twstock（依設定與可用性）｜本程式不使用 yfinance。"
        "技術訊號為規則化模型，實際交易仍需搭配流動性、市場環境、事件風險與個人風險承受度。"
    )


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None

    if _ctx is None:
        print("請使用 Streamlit 啟動此程式：")
        print("    streamlit run taiwan_quant_trader.py")
    else:
        main()
