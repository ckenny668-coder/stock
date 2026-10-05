# -*- coding: utf-8 -*-
"""
台股量化分析（本機版・修正版）
------------------------------------------------------------
執行：  python stock_agent_local.py            （預設 2313）
        python stock_agent_local.py 2330       （指定股票）
        python stock_agent_local.py 2330 2022-01-01

需要套件：pip install pandas numpy plotly requests
（不需要 FinMind 套件、pyarrow、yfinance、pandas_ta）

相對原版的修正：
1. 回測：訊號隔天「開盤」成交；停損遇跳空以開盤價成交；扣手續費與證交稅
2. 買進訊號改為「分數剛達標」，不再要求黃金交叉同日成立（交易次數太少）
3. 評分去除重複計算；現在的「建議」與回測使用同一套規則
4. RSI 連漲時正確為 100；法人資料缺值不再當 0
5. 法人買賣超單位由「股」換成「張」
6. 快取檔名含起迄日期，30 分鐘內有效
"""
import datetime
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FEE_RATE = 0.001425   # 手續費（買、賣各一次）
TAX_RATE = 0.003      # 證交稅（賣出）
BUY_TH = 5            # 買進分數門檻
SELL_TH = 4           # 賣出分數門檻


def finmind_get(dataset, stock_id, start, end, token=""):
    params = {"dataset": dataset, "data_id": str(stock_id),
              "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    r = requests.get(FINMIND_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(f"FinMind：{j.get('msg', '未知錯誤')}")
    return pd.DataFrame(j.get("data", []))


# ==========================================
# 1. Data Agent
# ==========================================
class DataAgent:
    def __init__(self, cache_dir="./stock_cache", token="", ttl_minutes=30):
        self.cache_dir = cache_dir
        self.token = token
        self.ttl = ttl_minutes * 60
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_data(self, ticker, start_date, end_date):
        cache = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
        if os.path.exists(cache) and (datetime.datetime.now().timestamp()
                                      - os.path.getmtime(cache)) < self.ttl:
            print(f"⚡ [DataAgent] 使用 30 分鐘內的本機快取：{ticker}")
            return pd.read_csv(cache, index_col="Date", parse_dates=True)

        print(f"🌐 [DataAgent] 向 FinMind 請求 {ticker} 日線與法人資料...")
        p = finmind_get("TaiwanStockPrice", ticker, start_date, end_date, self.token)
        if p.empty:
            raise ValueError(f"無法取得 {ticker} 的價格資料，請確認代號。")

        df = pd.DataFrame({
            "Date": pd.to_datetime(p["date"]),
            "Open": pd.to_numeric(p["open"], errors="coerce"),
            "High": pd.to_numeric(p["max"], errors="coerce"),
            "Low": pd.to_numeric(p["min"], errors="coerce"),
            "Close": pd.to_numeric(p["close"], errors="coerce"),
            "Volume": pd.to_numeric(p["Trading_Volume"], errors="coerce"),
        }).dropna().sort_values("Date").drop_duplicates("Date").set_index("Date")
        # 停牌／無成交（價格為 0）的日子剔除
        df = df[(df["Close"] > 0) & (df["Open"] > 0)]

        # 法人買賣超：股 → 張；抓不到時保留 NaN（不當 0）
        df["Institutional_Net"] = np.nan
        try:
            c = finmind_get("TaiwanStockInstitutionalInvestorsBuySell",
                            ticker, start_date, end_date, self.token)
            if not c.empty:
                g = c.groupby("date")
                net = (g["buy"].sum() - g["sell"].sum()) / 1000.0
                net.index = pd.to_datetime(net.index)
                df["Institutional_Net"] = net.reindex(df.index)
        except Exception as e:
            print(f"⚠️ [DataAgent] 法人資料讀取失敗，評分將略過此項：{e}")

        df.to_csv(cache)
        return df

    def fetch_price_only(self, ticker, start_date, end_date):
        """只抓日線（給大盤替身 ETF 用）。"""
        cache = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}_px.csv")
        if os.path.exists(cache) and (datetime.datetime.now().timestamp()
                                      - os.path.getmtime(cache)) < self.ttl:
            return pd.read_csv(cache, index_col="Date", parse_dates=True)
        p = finmind_get("TaiwanStockPrice", ticker, start_date, end_date, self.token)
        if p.empty:
            raise ValueError(f"無法取得 {ticker} 價格資料")
        df = pd.DataFrame({"Date": pd.to_datetime(p["date"]),
                           "Close": pd.to_numeric(p["close"], errors="coerce")}
                          ).dropna().sort_values("Date").drop_duplicates("Date").set_index("Date")
        df.to_csv(cache)
        return df

    def market_of(self, ticker):
        """回傳 '上市' / '上櫃' / ''（查不到）。"""
        try:
            info = finmind_get("TaiwanStockInfo", ticker, "2000-01-01",
                               datetime.date.today().strftime("%Y-%m-%d"), self.token)
            if not info.empty and "type" in info.columns:
                t = str(info["type"].iloc[0]).lower()
                return {"twse": "上市", "tpex": "上櫃"}.get(t, "")
        except Exception:
            pass
        return ""


# ==========================================
# 2. Analyst Agent
# ==========================================
class AnalystAgent:
    def process_indicators(self, df):
        df = df.copy()
        c = df["Close"]

        for n in (5, 20, 50):
            df[f"SMA_{n}"] = c.rolling(n).mean()

        df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = c.ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # RSI：沒有下跌時為 100
        d = c.diff()
        ag = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
        rsi = rsi.where(~((al == 0) & (ag > 0)), 100)
        df["RSI_14"] = rsi

        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
        df["BIAS_5"] = (c - df["SMA_5"]) / df["SMA_5"] * 100
        df["Inst_3D_Sum"] = df["Institutional_Net"].rolling(3).sum()   # 張；缺值為 NaN

        # ---------- 評分（每個概念只算一次）----------
        trend_up = (c > df["SMA_20"]) & (df["SMA_20"] > df["SMA_50"])
        trend_dn = (c < df["SMA_20"]) & (df["SMA_20"] < df["SMA_50"])
        i = lambda s: s.fillna(False).astype(bool).astype(int)

        df["Buy_Score"] = (i(trend_up) * 2 + i(c > df["SMA_5"]) + i(df["MACD_Hist"] > 0)
                           + i(df["RSI_14"] > 50) + i(df["Volume"] > df["Volume_MA20"])
                           + i(df["Inst_3D_Sum"] > 0))                       # 最高 7
        df["Sell_Score"] = (i(trend_dn) * 2 + i(c < df["SMA_5"]) + i(df["MACD_Hist"] < 0)
                            + i(df["RSI_14"] < 50) + i(df["Inst_3D_Sum"] < 0))  # 最高 6

        bs, ss = df["Buy_Score"], df["Sell_Score"]
        # 買進：分數「剛」達標（避免連續多天重複訊號）且站上月線
        df["Buy_Signal"] = ((bs >= BUY_TH) & (bs.shift(1) < BUY_TH) & (c > df["SMA_20"]) & (bs > ss))
        # 賣出：分數剛達標，或跌破 20 日線且 5MA 死亡交叉
        dead = (df["SMA_5"] < df["SMA_20"]) & (df["SMA_5"].shift(1) >= df["SMA_20"].shift(1))
        df["Sell_Signal"] = (((ss >= SELL_TH) & (ss.shift(1) < SELL_TH)) | dead) & (ss > bs)
        df["Buy_Signal"] = df["Buy_Signal"].fillna(False).astype(bool)
        df["Sell_Signal"] = df["Sell_Signal"].fillna(False).astype(bool)

        print("🧠 [AnalystAgent] 指標與評分計算完成。")
        return df


# ==========================================
# 3. Backtest Agent（隔日開盤成交／跳空停損／含成本）
# ==========================================
class BacktestAgent:
    def run_backtest(self, df, warmup_days=60, capital=1_000_000,
                     stop_loss=0.07, trailing_stop=0.10, lot_size=1000):
        d = df.iloc[warmup_days:].reset_index()
        cash, shares = float(capital), 0
        entry_price = entry_cost = highest = 0.0
        entry_date = None
        pending = None
        trades, eq = [], []

        def sell(date, price, reason):
            nonlocal cash, shares, entry_price, entry_cost, highest, entry_date
            gross = shares * price
            net = gross * (1 - FEE_RATE - TAX_RATE)
            cash += net
            trades.append({"進場日": entry_date, "出場日": date, "進場價": entry_price,
                           "出場價": price, "股數": shares, "損益": net - entry_cost,
                           "報酬%": (net - entry_cost) / entry_cost * 100, "原因": reason})
            shares, entry_price, entry_cost, highest, entry_date = 0, 0.0, 0.0, 0.0, None

        for _, r in d.iterrows():
            date, o, h, l, c = r["Date"], r["Open"], r["High"], r["Low"], r["Close"]
            exited_today = False

            # 1) 執行前一日訊號（開盤價）
            if pending == "buy" and shares == 0:
                qty = math.floor(cash / (o * (1 + FEE_RATE)) / lot_size) * lot_size
                if qty > 0:
                    cost = qty * o * (1 + FEE_RATE)
                    cash -= cost
                    shares, entry_price, entry_cost, highest, entry_date = qty, o, cost, o, date
            elif pending == "sell" and shares > 0:
                sell(date, o, "訊號賣出")
                exited_today = True
            pending = None

            # 2) 盤中停損／移動停利（跳空以開盤價成交）
            if shares > 0:
                stop = max(entry_price * (1 - stop_loss), highest * (1 - trailing_stop))
                if o <= stop:
                    sell(date, o, "停損／停利（跳空）")
                    exited_today = True
                elif l <= stop:
                    sell(date, stop, "停損／移動停利")
                    exited_today = True
                else:
                    highest = max(highest, h)

            # 3) 收盤後依訊號排定隔日動作
            if not exited_today:
                if shares == 0 and r["Buy_Signal"]:
                    pending = "buy"
                elif shares > 0 and r["Sell_Signal"]:
                    pending = "sell"

            eq.append({"Date": date, "Equity": cash + shares * c})

        last = d.iloc[-1]
        if shares > 0:
            sell(last["Date"], last["Close"], "回測結束平倉")
            eq[-1]["Equity"] = cash

        eq = pd.DataFrame(eq).set_index("Date")
        trades = pd.DataFrame(trades)

        # 買進持有（同樣用整張／同成本，第一天開盤買、最後收盤賣）
        o0, c1 = d["Open"].iloc[0], last["Close"]
        q0 = math.floor(capital / (o0 * (1 + FEE_RATE)) / lot_size) * lot_size
        bh_final = capital - q0 * o0 * (1 + FEE_RATE) + q0 * c1 * (1 - FEE_RATE - TAX_RATE)

        peak = eq["Equity"].cummax()
        mdd = ((eq["Equity"] - peak) / peak).min() * 100
        result = {
            "strategy_return": (eq["Equity"].iloc[-1] / capital - 1) * 100,
            "buy_hold_return": (bh_final / capital - 1) * 100,
            "max_drawdown": mdd,
            "n_trades": len(trades),
            "win_rate": (trades["損益"] > 0).mean() * 100 if len(trades) else np.nan,
            "trades": trades, "equity": eq,
        }
        print("📈 [BacktestAgent] 回測完成（隔日開盤成交、含手續費與證交稅）。")
        return result


# ==========================================
# 4. Visualization Agent
# ==========================================
class VisualizationAgent:
    def render(self, ticker, df, show_days=250, out_html=None, show=True):
        v = df.iloc[-show_days:].copy()
        x = v.index.strftime("%Y-%m-%d")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.50, 0.15, 0.18, 0.17],
                            subplot_titles=[f"{ticker} K線與進出場訊號", "成交量", "MACD", "RSI"])
        fig.add_trace(go.Candlestick(x=x, open=v["Open"], high=v["High"], low=v["Low"],
                                     close=v["Close"], name="K線",
                                     increasing_line_color="red", decreasing_line_color="green"),
                      row=1, col=1)
        for col, nm, color in [("SMA_5", "5MA", "orange"), ("SMA_20", "20MA", "blue"), ("SMA_50", "50MA", "purple")]:
            fig.add_trace(go.Scatter(x=x, y=v[col], name=nm, line=dict(width=1.2, color=color)), row=1, col=1)

        b, s = v[v["Buy_Signal"]], v[v["Sell_Signal"]]
        if len(b):
            fig.add_trace(go.Scatter(x=b.index.strftime("%Y-%m-%d"), y=b["Low"] * 0.98, mode="markers+text",
                                     name="買進", text=["買進"] * len(b), textposition="bottom center",
                                     textfont=dict(color="red"),
                                     marker=dict(symbol="triangle-up", size=10, color="red")), row=1, col=1)
        if len(s):
            fig.add_trace(go.Scatter(x=s.index.strftime("%Y-%m-%d"), y=s["High"] * 1.02, mode="markers+text",
                                     name="賣出", text=["賣出"] * len(s), textposition="top center",
                                     textfont=dict(color="green"),
                                     marker=dict(symbol="triangle-down", size=10, color="green")), row=1, col=1)

        vc = np.where(v["Close"] >= v["Open"], "red", "green")
        fig.add_trace(go.Bar(x=x, y=v["Volume"] / 1000, name="成交量(張)", marker_color=vc, opacity=0.5), row=2, col=1)
        mc = np.where(v["MACD_Hist"] >= 0, "red", "green")
        fig.add_trace(go.Bar(x=x, y=v["MACD_Hist"], name="柱狀體", marker_color=mc, opacity=0.5), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=v["MACD"], name="MACD", line=dict(width=1.2, color="blue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=v["MACD_Signal"], name="Signal", line=dict(width=1.2, color="orange")), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=v["RSI_14"], name="RSI14", line=dict(width=1.4, color="purple")), row=4, col=1)
        for y, dash, col in [(70, "dash", "red"), (50, "dot", "gray"), (30, "dash", "green")]:
            fig.add_hline(y=y, line_dash=dash, line_color=col, row=4, col=1)

        fig.update_layout(title=dict(text=f"<b>{ticker} 量化分析</b>", x=0.01), height=900,
                          hovermode="x unified", template="plotly_white",
                          xaxis_rangeslider_visible=False)
        fig.update_xaxes(type="category", nticks=12)
        if out_html:
            fig.write_html(out_html)
            if show:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(out_html))
        elif show:
            fig.show()
        return fig


# ==========================================
# 4b. 趨勢評分：價量警示／箱型突破／11分制／RS／Minervini／漲停
# ==========================================
def pv_warnings(df):
    """逐日價量警示文字。"""
    c, o, v = df["Close"], df["Open"], df["Volume"]
    ma5, ma20 = df["SMA_5"], df["SMA_20"]
    c1, c2, v1, v2 = c.shift(1), c.shift(2), v.shift(1), v.shift(2)
    out = []
    for i in range(len(df)):
        close, open_p, vol = c.iloc[i], o.iloc[i], v.iloc[i]
        pc, pv, m5, m20 = c1.iloc[i], v1.iloc[i], ma5.iloc[i], ma20.iloc[i]
        if any(pd.isna(x) for x in (pc, pv, m5, m20)):
            out.append("")
            continue
        w = []
        if close > m20 and pc <= m20:
            w.append("突破 20 日線")
        if close < m5 and pc >= m5:
            w.append("跌破 5 日線，短線防線失守，建議減碼")
        up, down, vup, vdn = close > pc, close < pc, vol > pv, vol < pv
        red, black = close > open_p, close < open_p
        if red and up:
            w.append("價漲量增，注意是否突破箱型高點（正式表態）" if vup
                     else "價漲量縮就難做，留意反彈／下影線" if vdn else "價漲量平")
        elif black and up:
            w.append("高檔震盪：外漲內跌、量增注意出貨風險" if vup else "高檔震盪：假陽線量縮，獲利回吐")
        elif down:
            w.append("價跌量增就難做" if vup else "價跌量縮" if vdn else "價跌量平可能續跌")
        else:
            w.append("價平量增" if vup else "價平量縮，等待主力" if vdn else "價平量平")
        p2, q2 = c2.iloc[i], v2.iloc[i]
        if pd.notna(p2) and pd.notna(q2):
            if close < p2 and vol > q2 and close < pc:
                w.append("低價夾量，連續創新低")
            if close > pc and vol < pv and close < p2:
                w.append("小量上漲、大量拉回")
        out.append("；".join(w))
    return pd.Series(out, index=df.index)


def trend_review(df, bench=None, market=""):
    """df 為 process_indicators 後資料；bench 為大盤替身日線（含 Close），可為 None。"""
    d = df.copy()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA150"] = d["Close"].rolling(150).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["BoxHigh"] = d["High"].rolling(20).max().shift(1)
    d["VolMA5"] = d["Volume"].rolling(5).mean()
    d["Breakout"] = (d["Close"] > d["BoxHigh"]) & (d["Volume"] > d["VolMA5"])

    cur, prev = d.iloc[-1], d.iloc[-2]
    hist_up = bool(pd.notna(cur["MACD_Hist"]) and pd.notna(prev["MACD_Hist"])
                   and cur["MACD_Hist"] > prev["MACD_Hist"])
    items = [("收盤 > MA5", bool(cur["Close"] > cur["SMA_5"]), 2),
             ("量能較前日增加", bool(cur["Volume"] > prev["Volume"]), 2),
             ("MACD 柱狀體上升", hist_up, 2),
             ("收盤 > MA20", bool(cur["Close"] > cur["SMA_20"]), 2),
             ("箱型突破（破前20日高且量>5日均量）", bool(cur["Breakout"]), 3)]
    score = sum(w for _, ok, w in items if ok)

    price = float(cur["Close"])
    ma50, ma150, ma200 = (float(cur[k]) if pd.notna(cur[k]) else np.nan for k in ("MA50", "MA150", "MA200"))
    bias50 = (price - ma50) / ma50 * 100 if pd.notna(ma50) and ma50 > 0 else np.nan

    rs_up, rs_note = None, "無基準資料，略過 RS"
    if bench is not None and len(bench):
        j = d[["Close"]].join(bench[["Close"]].rename(columns={"Close": "B"}), how="left")
        j["B"] = j["B"].ffill()
        j = j.dropna()
        if len(j) >= 6:
            rs = j["Close"] / j["B"]
            rs_up = bool(rs.iloc[-1] > rs.iloc[-5])
            rs_note = "RS 近5日向上（強於大盤）" if rs_up else "RS 近5日向下（弱於大盤）"

    v20 = d["Volume"].rolling(20).mean().iloc[-1]
    v_ratio = float(cur["Volume"] / v20) if pd.notna(v20) and v20 > 0 else np.nan

    if pd.notna(bias50) and bias50 > 30:
        status, reason = "🔴 轉為觀望", f"距 MA50 乖離 {bias50:.1f}%，高檔過度延伸，追價風險高"
    elif rs_up is False:
        status, reason = "🟡 轉為觀望", "相對強度落後大盤，防範假突破"
    elif score >= 7 or bool(cur["Breakout"]):
        status = "🟢 可以買進"
        reason = "箱型突破，正式表態" if bool(cur["Breakout"]) else "整理中多頭動能轉強"
    else:
        status, reason = "🟡 觀察中", "整理中，條件未齊"

    if pd.notna(ma50) and price > ma50:
        trend = "⚠️ 高檔極端延伸" if pd.notna(bias50) and bias50 > 30 else "🔥 多頭攻擊"
    else:
        trend = "☁️ 震盪整理"

    prev_close = float(prev["Close"])
    limit_up = round(prev_close * 1.1, 2)
    return dict(items=items, score=score, status=status, reason=reason, trend=trend,
                price=price, ma50=ma50, ma150=ma150, ma200=ma200, bias50=bias50,
                rs_up=rs_up, rs_note=rs_note, v_ratio=v_ratio, prev_close=prev_close,
                high=float(cur["High"]), limit_up=limit_up,
                touched=float(cur["High"]) >= limit_up - 0.05)


# ==========================================
# 5. Orchestrator
# ==========================================
class StockAnalysisOrchestrator:
    def __init__(self, token=""):
        self.data = DataAgent(token=token)
        self.analyst = AnalystAgent()
        self.bt = BacktestAgent()
        self.vis = VisualizationAgent()

    def print_trend_review(self, ticker, df, start_date, end_date):
        if len(df) < 60:
            print("⚠️ 資料不足 60 筆，略過趨勢評分。")
            return
        market = self.data.market_of(ticker)
        bsym = "006201" if market == "上櫃" else "0050"
        bname = "櫃買替身 ETF 006201" if market == "上櫃" else "大盤替身 ETF 0050"
        try:
            bench = self.data.fetch_price_only(bsym, start_date, end_date)
        except Exception as e:
            print(f"⚠️ 無法取得基準 {bsym}，略過 RS：{e}")
            bench = None
        tr = trend_review(df, bench, market)
        f = lambda x, fmt="{:.2f}": "-" if pd.isna(x) else fmt.format(x)

        print("\n" + "=" * 66)
        print(f"🎯 趨勢評分（11分制）｜{ticker}（{market or '市場未知'}）｜基準：{bname}")
        print(f"得分 {tr['score']}/11 | 建議：{tr['status']} | 趨勢：{tr['trend']}")
        print(f"原因：{tr['reason']}")
        print("-" * 66)
        for n, ok, w in tr["items"]:
            print(f"  {'✅' if ok else '❌'} {n}（{w}分）")
        print("-" * 66)
        print(f"現價 {tr['price']:.2f} | 50MA {f(tr['ma50'])} | 150MA {f(tr['ma150'])} | 200MA {f(tr['ma200'])}")
        print(f"50MA 乖離 {f(tr['bias50'], '{:.2f}%')} | 成交量倍率 {f(tr['v_ratio'], '{:.2f}x')} | {tr['rs_note']}")
        print("Minervini：股價>50MA " + ("✅" if pd.notna(tr["ma50"]) and tr["price"] > tr["ma50"] else "❌")
              + " | 50MA>200MA " + ("資料不足" if pd.isna(tr["ma200"]) else ("✅" if tr["ma50"] > tr["ma200"] else "❌"))
              + " | RS向上 " + ("資料不足" if tr["rs_up"] is None else ("✅" if tr["rs_up"] else "❌")))
        if pd.isna(tr["ma200"]):
            print("  （MA200 需 200 筆以上資料，可把開始日期往前拉）")
        print(f"漲停判斷：昨收 {tr['prev_close']:.2f} | 今高 {tr['high']:.2f} | 漲停價 {tr['limit_up']:.2f} | "
              f"{'✅ 觸及漲停' if tr['touched'] else '❌ 未觸及'}（簡化計算，未做升降單位進位）")

        w = df.tail(35)[["Open", "High", "Low", "Close", "Volume"]].copy()
        w["Volume"] = (w["Volume"] / 1000).round(0)
        w.columns = ["開", "高", "低", "收", "量(張)"]
        w["警示"] = pv_warnings(df).tail(35).values
        w.index = w.index.date
        print("\n近35日價量警示：")
        pd.set_option("display.max_colwidth", 80, "display.width", 200)
        print(w.to_string(float_format=lambda x: f"{x:,.2f}"))
        print("=" * 66)

    def run(self, ticker, start_date="2022-01-01", show=True):
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        print(f"\n🚀 開始分析 {ticker}（{start_date} ~ {end_date}）")

        df = self.analyst.process_indicators(self.data.fetch_data(ticker, start_date, end_date))
        if len(df) < 80:
            print(f"⚠️ 只有 {len(df)} 筆資料，指標與回測僅供參考。")
        res = self.bt.run_backtest(df)

        L = df.iloc[-1]
        inst = L["Inst_3D_Sum"]
        chip_ok = pd.notna(inst)
        bs, ss = int(L["Buy_Score"]), int(L["Sell_Score"])

        # 建議與回測使用同一套分數規則
        if L["Close"] < L["SMA_5"] and chip_ok and inst < 0:
            status = "🟢 偏空／防守（跌破5MA且法人賣超）"
        elif ss >= SELL_TH and ss > bs:
            status = "🟢 偏空（賣出分數達標）"
        elif bs >= BUY_TH and L["Close"] > L["SMA_20"]:
            status = "🔴 偏多（買進分數達標）" + ("，法人加碼" if chip_ok and inst > 0 else "")
        elif L["Close"] < L["SMA_5"]:
            status = "🟡 短線轉弱／觀望"
        else:
            status = "🟡 震盪整理／觀望"

        print("\n" + "=" * 66)
        print(f"📊 {ticker} | {df.index[-1]:%Y-%m-%d} | 收盤 {L['Close']:.2f} | 5MA {L['SMA_5']:.2f} | 20MA {L['SMA_20']:.2f} | 50MA {L['SMA_50']:.2f}")
        print(f"MACD {L['MACD']:.2f} | RSI14 {L['RSI_14']:.1f} | 5日乖離 {L['BIAS_5']:.2f}%")
        print("近3日法人買賣超：" + (f"{inst:,.0f} 張" if chip_ok else "無資料（評分略過此項，買進分數上限 6）"))
        print(f"買進分 {bs}/7 | 賣出分 {ss}/6 | 決策：{status}")
        print("-" * 66)
        t = res["trades"]
        wr = f"{res['win_rate']:.0f}%" if pd.notna(res["win_rate"]) else "-"
        print(f"回測：策略 {res['strategy_return']:.2f}% | 買進持有 {res['buy_hold_return']:.2f}% | "
              f"最大回撤 {res['max_drawdown']:.2f}% | 交易 {res['n_trades']} 次 | 勝率 {wr}")
        if res["n_trades"] < 10:
            print("⚠️ 交易次數少於 10 次，績效數字統計意義有限。")
        print("=" * 66)
        if len(t):
            print(t.tail(10).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
            print()

        self.print_trend_review(ticker, df, start_date, end_date)

        out = f"{ticker}_report.html"
        self.vis.render(ticker, df, out_html=out, show=show)
        print(f"🎨 圖表已存成 {out}")
        return df, res


def _pick_args(argv):
    """只接受「股票代號」與「YYYY-MM-DD」格式的參數，Jupyter 自帶的 -f xxx.json 會被忽略。"""
    import re
    tk, st = "2313", "2022-01-01"
    for a in argv[1:]:
        if re.fullmatch(r"\d{4,6}[A-Za-z]?", a):
            tk = a
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
            st = a
    return tk, st


if __name__ == "__main__":
    # 命令列：python stock_agent_local.py 2330 2022-01-01
    # Jupyter：直接改下面這行，或在儲存格寫  StockAnalysisOrchestrator().run("2330")
    tk, st = _pick_args(sys.argv)
    StockAnalysisOrchestrator().run(tk, st)
