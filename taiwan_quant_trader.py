import time
import datetime
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ==============================================================================
# 🤖 StockTraderAgent: 智慧股票分析與決策代理人
# ==============================================================================
class StockTraderAgent:
    # TWSE 回應快取：同一天的全市場資料只抓一次，批次掃描時大幅減少請求
    _twse_cache = {}

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, ticker, start_date="2022-01-01", sector_note="", bottleneck_note=""):
        self.ticker = str(ticker).strip()
        self.start_date = start_date
        self.sector_note = sector_note          # 步驟1：大趨勢/板塊
        self.bottleneck_note = bottleneck_note  # 步驟2：瓶頸節點
        self.market = None
        self.ticker_yf = None
        self.df = None
        self.df_rs = None
        self.score = 0
        self.bias_50ma = 0.0
        self.rs_slope_up = False
        self.agent_action = "HOLD"
        self.action_reasons = []

        self.capital_yi = None
        self.capital_flag = "資料不足"
        self.revenue_trend = None
        self.margin_trend = None
        self.bottleneck_verdict = "資料不足，無法判斷真偽"
        self.institutional_flow = None
        self.margin_balance_change = None
        self.stop_loss_price = None

    # --------------------------------------------------------------------------
    # 1. 抓取資料
    # --------------------------------------------------------------------------
    @staticmethod
    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def fetch_data(self):
        self.ticker_yf = f"{self.ticker}.TW"
        df = yf.download(self.ticker_yf, start=self.start_date, progress=False)
        self.market = "上市"

        if df.empty or len(df) < 10:
            self.ticker_yf = f"{self.ticker}.TWO"
            df = yf.download(self.ticker_yf, start=self.start_date, progress=False)
            self.market = "上櫃"

        if df.empty or len(df) < 10:
            raise ValueError(f"無法找到股票代碼 {self.ticker} 的足夠資料（至少需要 10 筆）。")

        df = self._flatten(df)
        df.index = pd.to_datetime(df.index)
        df["Volume"] = df["Volume"] / 1000  # 轉為張數

        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        df["Close_shift1"] = df["Close"].shift(1)
        df["Close_shift2"] = df["Close"].shift(2)
        df["Volume_shift1"] = df["Volume"].shift(1)
        df["Volume_shift2"] = df["Volume"].shift(2)

        self.df = df
        print(f"✅ [Agent] 成功獲取 {self.ticker} ({self.market}) 行情數據。共 {len(df)} 筆交易紀錄。")

    # --------------------------------------------------------------------------
    # 2. 價量與警示檢查
    # --------------------------------------------------------------------------
    def _check_warnings(self, row):
        warnings_list = []
        try:
            close = float(row["Close"])
            open_p = float(row["Open"])
            volume = float(row["Volume"])
            ma5 = float(row["MA5"]) if pd.notna(row["MA5"]) else None
            ma20 = float(row["MA20"]) if pd.notna(row["MA20"]) else None
            close_shift1 = float(row["Close_shift1"]) if pd.notna(row["Close_shift1"]) else None
            close_shift2 = float(row["Close_shift2"]) if pd.notna(row["Close_shift2"]) else None
            volume_shift1 = float(row["Volume_shift1"]) if pd.notna(row["Volume_shift1"]) else None
            volume_shift2 = float(row["Volume_shift2"]) if pd.notna(row["Volume_shift2"]) else None
        except (ValueError, TypeError):
            return ["數據轉換錯誤"]

        if None in [close_shift1, volume_shift1, ma5, ma20]:
            return warnings_list

        if close > ma20 and close_shift1 <= ma20:
            warnings_list.append("突破 20 日線")
        if close < ma5 and close_shift1 >= ma5:
            warnings_list.append("跌破 5 日線，短線防線失守，建議減碼")

        is_price_up = close > close_shift1
        is_price_down = close < close_shift1
        is_vol_up = volume > volume_shift1
        is_vol_down = volume < volume_shift1
        is_red_k = close > open_p
        is_black_k = close < open_p

        if is_red_k and is_price_up:
            if is_vol_up:
                warnings_list.append("價漲量增，注意主力突破「箱型」高點，才算正式表態")
            elif is_vol_down:
                warnings_list.append("價漲量縮，屬於弱勢反彈")
            else:
                warnings_list.append("價漲量平")
        elif is_black_k and is_price_up:
            if is_vol_up:
                warnings_list.append("高檔震盪：出貨或換手？(量增注意高檔拉回風險)")
            else:
                warnings_list.append("高檔震盪：假陽線量縮，多頭獲利回吐震盪")
        elif is_price_down:
            if is_vol_up:
                warnings_list.append("價跌量增，短線賣壓強烈")
            elif is_vol_down:
                warnings_list.append("價跌量縮")
            else:
                warnings_list.append("價跌量平可能續跌")

        if close_shift2 and volume_shift2:
            if close < close_shift2 and volume > volume_shift2 and close < close_shift1:
                warnings_list.append("低價夾量連續三天創新低")
            if close > close_shift1 and volume < volume_shift1 and close < close_shift2:
                warnings_list.append("小量上漲、大量拉回")

        return warnings_list

    # --------------------------------------------------------------------------
    # 3. 核心決策與推理引擎
    # --------------------------------------------------------------------------
    def analyze_and_decide(self):
        if self.df is None or len(self.df) < 30:
            raise ValueError("資料不足，無法進行技術面分析。")

        self.df["warnings"] = self.df.apply(self._check_warnings, axis=1)

        df2 = self.df.iloc[-150:].copy()
        df2["MA5"] = ta.sma(df2["Close"], length=5)
        df2["MA20"] = ta.sma(df2["Close"], length=20)

        macd_df = ta.macd(df2["Close"])
        if macd_df is not None and not macd_df.empty:
            df2 = pd.concat([df2, macd_df], axis=1)

        df2["Box_High"] = df2["High"].rolling(20).max().shift(1)
        df2["Vol_MA5"] = df2["Volume"].rolling(5).mean()
        df2["Is_Breakout"] = (df2["Close"] > df2["Box_High"]) & (df2["Volume"] > df2["Vol_MA5"])

        curr, prev = df2.iloc[-1], df2.iloc[-2]

        self.score = 0
        self.score += 2 if curr["Close"] > curr["MA5"] else 0
        self.score += 2 if curr["Volume"] > prev["Volume"] else 0
        self.score += 2 if curr.get("MACDh_12_26_9", 0) > prev.get("MACDh_12_26_9", 0) else 0
        self.score += 2 if curr["Close"] > curr["MA20"] else 0
        self.score += 3 if bool(curr.get("Is_Breakout", False)) else 0

        # 相對強度 (RS)
        benchmark = "^TWII" if self.market == "上市" else "^TWOII"
        start_date = self.df.index[0].strftime("%Y-%m-%d")
        end_date = (self.df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        benchmark_df = yf.download(benchmark, start=start_date, end=end_date, progress=False)
        if benchmark_df.empty and benchmark != "^TWII":
            benchmark_df = yf.download("^TWII", start=start_date, end=end_date, progress=False)
        benchmark_df = self._flatten(benchmark_df)

        df_rs = pd.DataFrame(index=self.df.index)
        df_rs["Ticker_Close"] = self.df["Close"]
        df_rs["Benchmark_Close"] = benchmark_df["Close"]
        df_rs = df_rs.ffill().dropna()

        if len(df_rs) >= 50:
            df_rs["RS_Line"] = df_rs["Ticker_Close"] / df_rs["Benchmark_Close"]
            df_rs["MA50"] = df_rs["Ticker_Close"].rolling(window=50).mean()

            current_price = df_rs["Ticker_Close"].iloc[-1]
            ma50 = df_rs["MA50"].iloc[-1]
            self.rs_slope_up = df_rs["RS_Line"].iloc[-1] > df_rs["RS_Line"].iloc[-5]
            self.bias_50ma = (
                ((current_price - ma50) / ma50) * 100 if pd.notna(ma50) and ma50 != 0 else 0.0
            )

        self.df_rs = df_rs

        # RSI 過熱判斷
        rsi5 = ta.rsi(self.df["Close"], length=5)
        rsi10 = ta.rsi(self.df["Close"], length=10)

        curr_rsi5 = rsi5.iloc[-1] if rsi5 is not None and not rsi5.dropna().empty else 50

        rsi10_over_90_cnt = 0
        if rsi10 is not None and not rsi10.dropna().empty:
            for val in rsi10.tail(10).values[::-1]:  # 由最新往回數
                if pd.notna(val) and val >= 90:
                    rsi10_over_90_cnt += 1
                else:
                    break

        # 動作推理
        self.action_reasons = []
        last_close = self.df["Close"].iloc[-1]

        if curr_rsi5 >= 90 or rsi10_over_90_cnt >= 2:
            self.agent_action = "🚨 強烈建議賣出 (SELL)"
            if curr_rsi5 >= 90:
                self.action_reasons.append(f"RSI(5)={curr_rsi5:.1f} 超過 90，短線極端過熱")
            if rsi10_over_90_cnt >= 2:
                self.action_reasons.append(
                    f"RSI(10) 超過 90 持續 {rsi10_over_90_cnt} 天，面臨急跌洗盤風險"
                )
        elif last_close < self.df["MA5"].iloc[-1] and self.df["Close_shift1"].iloc[-1] >= self.df["MA5"].iloc[-1]:
            self.agent_action = "⚠️ 建議減碼賣出 (REDUCE)"
            self.action_reasons.append("跌破 5 日線，短線防線失守")
        elif self.bias_50ma > 30.0:
            self.agent_action = "🔴 觀望 (WAIT)"
            self.action_reasons.append(f"50MA 乖離率達 {self.bias_50ma:.1f}%，高檔延伸過度")
        elif not self.rs_slope_up:
            self.agent_action = "🟡 觀望 (WAIT)"
            self.action_reasons.append("相對強度 (RS) 落後大盤，防範假突破")
        elif self.score >= 7 or bool(curr.get("Is_Breakout", False)):
            self.agent_action = "🟢 建議買進 (BUY)"
            reason = "正式突破箱型表態" if bool(curr.get("Is_Breakout", False)) else "多頭動能轉強"
            self.action_reasons.append(f"{reason}，技術面評分: {int(self.score)}/11")
        else:
            self.agent_action = "🟡 繼續觀察 (OBSERVE)"
            self.action_reasons.append(f"盤整階段，買進得分為 {int(self.score)}/11")

        self.stop_loss_price = round(float(last_close) * 0.90, 2)

    # --------------------------------------------------------------------------
    # 4. 【步驟3】股本大小檢查
    # --------------------------------------------------------------------------
    def check_capital_size(self):
        try:
            info = yf.Ticker(self.ticker_yf).info
            shares_out = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")

            if shares_out:
                self.capital_yi = round((shares_out * 10) / 1e8, 1)  # 面額 10 元
                if 5 <= self.capital_yi <= 30:
                    self.capital_flag = (
                        f"股本約 {self.capital_yi} 億元，符合「小龍頭」區間 (5~30億)，EPS 彈性佳"
                    )
                elif self.capital_yi < 5:
                    self.capital_flag = (
                        f"股本約 {self.capital_yi} 億元，屬於小型股，注意流動性與波動風險"
                    )
                else:
                    self.capital_flag = (
                        f"股本約 {self.capital_yi} 億元，屬於中大型股，股性相對穩健"
                    )
            else:
                self.capital_flag = "yfinance 無法取得股本資料，請至公開資訊觀測站查詢"
        except Exception as e:
            self.capital_flag = f"股本資料擷取異常：{e}"

    # --------------------------------------------------------------------------
    # 5. 【步驟4】真偽篩選 (營收與毛利率)
    # --------------------------------------------------------------------------
    def check_bottleneck_authenticity(self):
        try:
            fin = yf.Ticker(self.ticker_yf).quarterly_financials
            if fin is not None and not fin.empty:
                rev_key = next((k for k in fin.index if "Revenue" in k), None)
                gp_key = next((k for k in fin.index if "Gross Profit" in k), None)

                if rev_key:
                    revenue = fin.loc[rev_key].dropna()
                    self.revenue_trend = revenue

                    if gp_key:
                        gross_profit = fin.loc[gp_key].dropna()
                        margin = (gross_profit / revenue * 100).round(2).dropna()
                        self.margin_trend = margin

                        if len(margin) >= 2 and len(revenue) >= 2:
                            is_margin_up = margin.iloc[0] > margin.iloc[1]
                            is_revenue_up = revenue.iloc[0] > revenue.iloc[1]
                            if is_margin_up and is_revenue_up:
                                self.bottleneck_verdict = "營收與毛利率同步走升，符合『真瓶頸具備定價權』特質"
                            elif is_revenue_up and not is_margin_up:
                                self.bottleneck_verdict = "營收成長但毛利率下滑，需注意是否為低價搶單"
                            else:
                                self.bottleneck_verdict = "營收與毛利率表現平淡，瓶頸效益尚未顯現"
                        else:
                            self.bottleneck_verdict = "財報數據期數不足"
                    else:
                        self.bottleneck_verdict = "查無毛利數據"
                else:
                    self.bottleneck_verdict = "查無季度營收數據"
            else:
                self.bottleneck_verdict = "查無財報資料，請至 MOPS 公開資訊觀測站確認"
        except Exception as e:
            self.bottleneck_verdict = f"財報查詢失敗：{e}"

    # --------------------------------------------------------------------------
    # 6. 【步驟5】籌碼面 (TWSE)
    # --------------------------------------------------------------------------
    @staticmethod
    def _to_int(s):
        try:
            return int(str(s).replace(",", "").replace("+", "").strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _get_twse(cls, url):
        """帶快取的 TWSE 請求；失敗回傳 None。"""
        if url in cls._twse_cache:
            return cls._twse_cache[url]
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=8).json()
        except Exception:
            resp = None
        cls._twse_cache[url] = resp
        return resp

    @staticmethod
    def _find_idx(fields, *keywords, exclude=()):
        """依欄位名稱找索引（所有關鍵字都需出現）。"""
        for i, f in enumerate(fields):
            if all(k in f for k in keywords) and not any(x in f for x in exclude):
                return i
        return None

    def _parse_t86(self, resp):
        """解析三大法人買賣超（依欄位名稱定位，避免索引錯位）。"""
        if not resp or resp.get("stat") != "OK":
            return None
        fields = resp.get("fields", [])
        rows = resp.get("data", [])
        row = next((d for d in rows if str(d[0]).strip() == self.ticker), None)
        if row is None:
            return None

        idx_foreign = next(
            (i for i, f in enumerate(fields)
             if "買賣超" in f and f.startswith(("外陸資", "外資"))
             and not f.startswith("外資自營商")),
            None,
        )
        idx_trust = next(
            (i for i, f in enumerate(fields) if f.startswith("投信") and "買賣超" in f),
            None,
        )
        idx_dealer = next(
            (i for i, f in enumerate(fields) if f.startswith("自營商買賣超")),
            None,
        )
        idx_total = next(
            (i for i, f in enumerate(fields) if f.startswith("三大法人買賣超")),
            None,
        )

        def val(i):
            return row[i] if i is not None and i < len(row) else "N/A"

        return (
            f"外資: {val(idx_foreign)} | 投信: {val(idx_trust)} | "
            f"自營商: {val(idx_dealer)} | 合計: {val(idx_total)} (股)"
        )

    def _parse_margin(self, resp):
        """解析融資融券，兼容舊版 data 與新版 tables 格式。"""
        if not resp or resp.get("stat") != "OK":
            return None

        candidate_rows = []
        if "tables" in resp:
            for t in resp["tables"]:
                candidate_rows.extend(t.get("data", []) or [])
        if "data" in resp:
            candidate_rows.extend(resp["data"] or [])

        row = next((d for d in candidate_rows if str(d[0]).strip() == self.ticker), None)
        if row is None or len(row) < 13:
            return None

        # 融資: [5]前日餘額 [6]今日餘額；融券: [11]前日餘額 [12]今日餘額
        m_prev, m_now = self._to_int(row[5]), self._to_int(row[6])
        s_prev, s_now = self._to_int(row[11]), self._to_int(row[12])

        def diff(now, prev):
            return f"{now - prev:+,}" if now is not None and prev is not None else "N/A"

        return (
            f"融資增減: {diff(m_now, m_prev)} 張 (餘額 {m_now:,}) | "
            f"融券增減: {diff(s_now, s_prev)} 張 (餘額 {s_now:,})"
            if None not in (m_now, s_now)
            else f"融資增減: {diff(m_now, m_prev)} 張 | 融券增減: {diff(s_now, s_prev)} 張"
        )

    def check_chip_data(self):
        if self.market != "上市":
            note = "上櫃股票請至櫃買中心 (TPEx) 網站查詢"
            self.institutional_flow = note
            self.margin_balance_change = note
            return

        today = datetime.date.today()
        for i in range(7):  # 往前最多找 7 天，涵蓋連假
            d = today - datetime.timedelta(days=i)
            if d.weekday() >= 5:  # 跳過週末
                continue
            qd = d.strftime("%Y%m%d")

            if not self.institutional_flow:
                url_t86 = (
                    f"https://www.twse.com.tw/rwd/zh/fund/T86"
                    f"?response=json&date={qd}&selectType=ALLBUT0999"
                )
                parsed = self._parse_t86(self._get_twse(url_t86))
                if parsed:
                    self.institutional_flow = f"日期 {qd} | {parsed}"

            if not self.margin_balance_change:
                url_m = (
                    f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
                    f"?response=json&date={qd}&selectType=ALL"
                )
                parsed = self._parse_margin(self._get_twse(url_m))
                if parsed:
                    self.margin_balance_change = f"日期 {qd} | {parsed}"

            if self.institutional_flow and self.margin_balance_change:
                break

        if not self.institutional_flow:
            self.institutional_flow = "近幾日無資料（可能為非交易日或 API 限制）"
        if not self.margin_balance_change:
            self.margin_balance_change = "近幾日無資料（可能為非交易日或 API 限制）"

    # --------------------------------------------------------------------------
    # 7. 互動式 K 線圖
    # --------------------------------------------------------------------------
    def plot_chart(self):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=(f"{self.ticker} 價量 K 線圖", "成交量 (張)"),
            row_width=[0.2, 0.7],
        )

        fig.add_trace(go.Candlestick(
            x=self.df.index, open=self.df["Open"], high=self.df["High"],
            low=self.df["Low"], close=self.df["Close"], name="K線",
            hovertext=self.df["warnings"].apply(lambda x: "<br>".join(x) if x else "無警示"),
            increasing_line_color="red", decreasing_line_color="green",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.df.index, y=self.df["MA5"], line=dict(color="#1f77b4", width=1.5), name="MA5"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df["MA20"], line=dict(color="#ff7f0e", width=1.5), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df["MA50"], line=dict(color="#2ca02c", width=1.5), name="MA50"), row=1, col=1)

        colors = ["red" if c >= o else "green" for c, o in zip(self.df["Close"], self.df["Open"])]
        fig.add_trace(go.Bar(x=self.df.index, y=self.df["Volume"], name="成交量", marker_color=colors), row=2, col=1)

        fig.update_layout(
            title=f"<b>{self.ticker} ({self.market}) - StockTraderAgent 決策圖表</b>",
            xaxis_rangeslider_visible=False, template="plotly_white",
            height=700, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.show()

    # --------------------------------------------------------------------------
    # 8. 文字簡報
    # --------------------------------------------------------------------------
    def report(self):
        print("\n" + "=" * 65)
        print(f"🤖 【StockTraderAgent】 台股選股六步驟決策報告 - {self.ticker}")
        print("=" * 65)

        print("\n【步驟1】大趨勢 / 板塊")
        print(f"  {self.sector_note if self.sector_note else '（未填寫）'}")

        print("\n【步驟2】供應鏈瓶頸節點")
        print(f"  {self.bottleneck_note if self.bottleneck_note else '（未填寫）'}")

        print("\n【步驟3】股本大小 / 小龍頭條件")
        print(f"  {self.capital_flag}")

        print("\n【步驟4】真偽篩選 (營收與毛利率趨勢)")
        print(f"  結論：{self.bottleneck_verdict}")

        print("\n【步驟5】籌碼面 (三大法人與融資融券)")
        print(f"  法人籌碼：{self.institutional_flow}")
        print(f"  資券變化：{self.margin_balance_change}")

        print("\n【步驟6】進出場策略 (技術面診斷)")
        print(f"  最新收盤價: {self.df['Close'].iloc[-1]:.2f}")
        print(f"  50MA 乖離率: {self.bias_50ma:.2f}%")
        print(f"  RS 指標趨勢: {'向上 (領先大盤)' if self.rs_slope_up else '向下 (落後大盤)'}")
        print(f"  技術面評分: {int(self.score)} / 11")
        print(f"  參考停損價 (-10%): {self.stop_loss_price}")
        print(f"  🎯 代理人最終決策: {self.agent_action}")
        print("  💡 決策依據:")
        for r in self.action_reasons:
            print(f"     • {r}")
        print("=" * 65 + "\n")


# ==============================================================================
# 🚀 批次掃描追蹤清單 + 自動繪製「買進 (BUY)」標的圖表
# ==============================================================================
# 步驟1 / 步驟2 備註：{代碼: (大趨勢/板塊, 瓶頸節點)}，沒列到的會顯示「（未填寫）」
notes_map = {
    "6213": ("AI 伺服器 / 高階銅箔基板 (CCL)", "高頻高速 CCL 材料供應"),
    "2303": ("晶圓代工成熟製程，受惠車用與高階封裝", "28nm/22nm 特殊製程產能"),
    # "XXXX": ("大趨勢", "瓶頸節點"),
}

watch_list = [
    "3711", "6257", "4977", "6588", "6213", "6191", "4958", "6213", "8358", "2303",
    "5347", "4566", "2313", "6285", "3090", "6568", "3535", "2467", "2337", "2327",
    "2360", "3035",
]


def run_scan(tickers, notes=None, sleep_sec=0.5, plot_buys=True):
    notes = notes or {}
    unique_list = list(dict.fromkeys(tickers))  # 去重並保持順序

    summary_results = []
    buy_agents = []

    print(f"🔍 開始分析追蹤清單，共計 {len(unique_list)} 檔股票...\n")

    for i, ticker in enumerate(unique_list, 1):
        print(f"[{i}/{len(unique_list)}] 正在分析 {ticker}...")
        try:
            sector, bottleneck = notes.get(ticker, ("", ""))
            agent = StockTraderAgent(ticker, sector_note=sector, bottleneck_note=bottleneck)

            agent.fetch_data()
            agent.check_capital_size()
            agent.check_bottleneck_authenticity()
            agent.analyze_and_decide()
            agent.check_chip_data()
            agent.report()

            summary_results.append({
                "股票代碼": ticker,
                "市場": agent.market,
                "板塊": sector or "（未填寫）",
                "瓶頸節點": bottleneck or "（未填寫）",
                "當前價格": f"{agent.df['Close'].iloc[-1]:.2f}",
                "技術評分": f"{int(agent.score)}/11",
                "50MA乖離率": f"{agent.bias_50ma:.1f}%",
                "RS趨勢": "向上" if agent.rs_slope_up else "落後",
                "股本(億)": agent.capital_yi if agent.capital_yi else "N/A",
                "決策指引": agent.agent_action,
            })

            if "BUY" in agent.agent_action or "買進" in agent.agent_action:
                buy_agents.append(agent)

        except Exception as e:
            print(f"❌ 分析 {ticker} 時發生錯誤: {e}\n")
            continue
        finally:
            time.sleep(sleep_sec)  # 避免請求過於頻繁

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        print("\n" + "★" * 70)
        print("📋 【StockTraderAgent】 追蹤清單綜合決策總表")
        print("★" * 70)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(summary_df.to_string(index=False))
        print("★" * 70 + "\n")

    if plot_buys:
        if buy_agents:
            print(f"🟢 找到 {len(buy_agents)} 檔符合買進條件的股票，開始繪製圖表：")
            for b in buy_agents:
                print(f"📉 繪製 {b.ticker} ({b.market}) K線圖中...")
                b.plot_chart()
        else:
            print("🟡 今日追蹤清單中無符合「買進 (BUY)」條件的標的，無繪製圖表。")

    return summary_results, buy_agents


if __name__ == "__main__":
    run_scan(watch_list, notes=notes_map)
