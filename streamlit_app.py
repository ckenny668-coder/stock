"""
Streamlit App - Stan Weinstein 階段分析
（Stage 3 頂部 + 耗盡缺口偵測 + 大戶持股籌碼）
=====================================================================
互動輸入：
  - 個股代號（美股如 AAPL、台股如 2330.TW）
  - 起訖日期
  - 分析參數（均線、延伸乖離率、跳空、爆量）
  - 台股籌碼：大戶持股比例（副圖）+ 籌碼訊號

大戶持股資料來源（可同時使用，依序合併）：
  1. FinMind 股權分散表（完整歷史，需 backer/sponsor 會員 token；取得的資料會存進 holders_cache/）
  2. 手動貼上（例如直接從你的籌碼 K 線 app 複製每週表格，優先於其他來源）

執行方式：
    pip install streamlit yfinance pandas numpy matplotlib requests
    streamlit run streamlit_app.py
"""

import os
import re
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import yfinance as yf

# ----------------------------------------------------------------------
# 中文字型設定（依作業系統嘗試常見的中文字型，找不到就用系統預設）
# ----------------------------------------------------------------------
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Noto)
    "C:/Windows/Fonts/msjh.ttc",     # Windows 微軟正黑體
    "C:/Windows/Fonts/msyh.ttc",     # Windows 微軟雅黑
    "C:/Windows/Fonts/simhei.ttf",   # Windows 黑體
    "/System/Library/Fonts/PingFang.ttc",  # macOS
]
for _path in _CJK_CANDIDATES:
    if os.path.exists(_path):
        try:
            fm.fontManager.addfont(_path)
            _name = fm.FontProperties(fname=_path).get_name()
            plt.rcParams["font.sans-serif"] = [_name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


# ----------------------------------------------------------------------
# 股價資料
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).astype("datetime64[ns]")
    return df


# ----------------------------------------------------------------------
# 大戶持股資料
#   內部統一格式：DataFrame(date, ge400, ge600, ge800, ge1000)
#   ge1000 = 持股 1000 張以上的比例（%），其餘類推
# ----------------------------------------------------------------------
THRESHOLDS = (400, 600, 800, 1000)
CACHE_DIR = Path("holders_cache")


def is_tw_ticker(ticker: str) -> bool:
    return ticker.upper().endswith((".TW", ".TWO"))


def stock_id_of(ticker: str) -> str:
    return ticker.split(".")[0].strip()


def _level_lower_bound(label: str):
    """從 FinMind 的持股分級字串（例如 '1000-5000'、'more than 1,000,001'）取出下限股數。"""
    s = str(label)
    low = s.lower()
    if "total" in low or "合計" in s or "差異" in s:
        return None
    nums = re.findall(r"\d[\d,]*", s)
    if not nums:
        return None
    return int(nums[0].replace(",", ""))


def fetch_finmind_history(stock_id: str, start: dt.date, end: dt.date, token: str) -> pd.DataFrame:
    resp = requests.get(
        "https://api.finmindtrade.com/api/v4/data",
        params={
            "dataset": "TaiwanStockHoldingSharesPer",
            "data_id": stock_id,
            "start_date": str(start),
            "end_date": str(end),
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    payload = resp.json()
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("msg", f"HTTP {resp.status_code}"))
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["date", *[f"ge{t}" for t in THRESHOLDS]])
    df["lb"] = df["HoldingSharesLevel"].map(_level_lower_bound)
    df = df.dropna(subset=["lb"])
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for d, g in df.groupby("date"):
        row = {"date": d}
        for t in THRESHOLDS:
            row[f"ge{t}"] = float(g.loc[g["lb"] > t * 1000, "percent"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _cache_path(stock_id: str) -> Path:
    return CACHE_DIR / f"{stock_id}.csv"


def read_cache(stock_id: str) -> pd.DataFrame:
    p = _cache_path(stock_id)
    if not p.exists():
        return pd.DataFrame(columns=["date", *[f"ge{t}" for t in THRESHOLDS]])
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def write_cache(stock_id: str, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        merged = pd.concat([read_cache(stock_id), df]).drop_duplicates("date", keep="last")
        merged.sort_values("date").to_csv(_cache_path(stock_id), index=False)
    except Exception:
        pass  # 存不了快取不影響分析


def parse_manual_holders(text: str, today: dt.date | None = None) -> pd.DataFrame:
    """
    解析手動貼上的資料。每行：日期 + 空白/逗號 + 大戶持股%（取該行日期後的第一個數字）。
    支援 2026-10-16、2026/10/16、10/16（省略年份時取不晚於今天的最近一次）。
    可以直接從籌碼 K 線 app 的表格複製貼上。
    """
    today = today or dt.date.today()
    rows = []
    pattern = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2})[^\d]+(\d+(?:\.\d+)?)")
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        ds, val = m.group(1), float(m.group(2))
        try:
            if re.match(r"\d{4}", ds):
                d = pd.to_datetime(ds.replace("/", "-")).date()
            else:
                mm, dd = (int(x) for x in ds.split("/"))
                d = dt.date(today.year, mm, dd)
                if d > today:
                    d = dt.date(today.year - 1, mm, dd)
        except ValueError:
            continue
        if 0 < val <= 100:
            rows.append({"date": pd.Timestamp(d), "big_pct": val})
    return pd.DataFrame(rows, columns=["date", "big_pct"])


def build_holders_series(
    stock_id: str,
    thr_lots: int,
    start: dt.date,
    end: dt.date,
    token: str = "",
    manual_text: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """合併各來源，回傳 (DataFrame[date, big_pct, chg_1w, chg_4w], 狀態訊息)。"""
    notes: list[str] = []
    col = f"ge{thr_lots}"
    frames = [read_cache(stock_id)]

    if token.strip():
        try:
            fm_df = fetch_finmind_history(stock_id, start - dt.timedelta(days=60), end, token.strip())
            if fm_df.empty:
                notes.append("FinMind 沒有回傳此股票的股權分散資料。")
            else:
                frames.append(fm_df)
                write_cache(stock_id, fm_df)
                notes.append(f"FinMind：取得 {len(fm_df)} 週資料。")
        except Exception as e:  # noqa: BLE001
            notes.append(f"FinMind 讀取失敗：{e}")

    parts = [f for f in frames if not f.empty]
    series = (
        pd.concat(parts)[["date", col]].rename(columns={col: "big_pct"})
        if parts else pd.DataFrame(columns=["date", "big_pct"])
    )

    manual = parse_manual_holders(manual_text) if manual_text.strip() else pd.DataFrame()
    if not manual.empty:
        series = pd.concat([series, manual])  # 手動資料優先（keep="last"）
        notes.append(f"手動資料：{len(manual)} 筆。")

    if series.empty:
        return pd.DataFrame(columns=["date", "big_pct", "chg_1w", "chg_4w"]), notes

    series["date"] = pd.to_datetime(series["date"]).astype("datetime64[ns]")
    series = series.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return add_holder_changes(series), notes


def add_holder_changes(w: pd.DataFrame) -> pd.DataFrame:
    """週增減與 4 週累計增減；資料斷週（間隔過大）時不硬算，避免誇大變化。"""
    w = w.sort_values("date").reset_index(drop=True).copy()
    gap_days = w["date"].diff().dt.days
    w["chg_1w"] = w["big_pct"].diff().where(gap_days.between(1, 10))

    prev = w[["date", "big_pct"]].rename(columns={"date": "pdate", "big_pct": "prev_pct"})
    target = w[["date"]].copy()
    target["target"] = target["date"] - pd.Timedelta(days=28)
    merged = pd.merge_asof(
        target.sort_values("target"), prev.sort_values("pdate"),
        left_on="target", right_on="pdate",
        direction="nearest", tolerance=pd.Timedelta(days=6),
    )
    w["chg_4w"] = w["big_pct"].values - merged["prev_pct"].values
    return w


# ----------------------------------------------------------------------
# Weinstein 分析核心 + 籌碼訊號
# ----------------------------------------------------------------------
def weinstein_analysis(
    df: pd.DataFrame,
    ma_window: int = 150,
    extension_threshold: float = 0.30,
    gap_atr_mult: float = 2.0,
    atr_window: int = 14,
    volume_mult: float = 1.7,
    volume_days: int = 3,
    holders: pd.DataFrame | None = None,
    surge_pts: float = 3.0,
    dist_pts: float = 1.0,
    require_chips: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out["ma30w"] = out["close"].rolling(ma_window, min_periods=max(ma_window // 3, 5)).mean()
    out["ma30w_slope"] = out["ma30w"].diff(10)

    out["deviation_pct"] = (out["close"] - out["ma30w"]) / out["ma30w"]
    out["is_extended"] = out["deviation_pct"] > extension_threshold

    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(atr_window).mean()

    out["gap"] = out["open"] - prev_close
    out["gap_up"] = out["gap"] > gap_atr_mult * out["atr"]

    # 爆量：成交量 > 20 日均量 * volume_mult；缺口當日起算往後 volume_days 天內有爆量即可
    vol_ma = out["volume"].rolling(20).mean()
    out["volume_spike"] = out["volume"] > volume_mult * vol_ma
    volume_spike_nearby = (
        out["volume_spike"][::-1].rolling(volume_days, min_periods=1).max()[::-1].astype(bool)
    )
    out["exhaustion_gap"] = out["gap_up"] & out["is_extended"] & volume_spike_nearby

    # ---- 大戶持股：週資料併入日線 ----
    # 股權分散表以週五為基準日，約週末才公布，所以從「基準日 + 3 天」（週一）起才視為可用，避免偷看未來。
    for c in ("holders_date", "big_pct", "big_chg_1w", "big_chg_4w"):
        out[c] = np.nan
    out["holders_date"] = pd.NaT
    if holders is not None and not holders.empty:
        h = holders.copy()
        h["avail"] = h["date"] + pd.Timedelta(days=3)
        left = out[["close"]].copy()
        left.index.name = "idx"
        left = left.reset_index()
        left["idx"] = left["idx"].astype("datetime64[ns]")
        h["avail"] = h["avail"].astype("datetime64[ns]")
        merged = pd.merge_asof(
            left.sort_values("idx"),
            h.rename(columns={"date": "holders_date_", "chg_1w": "c1", "chg_4w": "c4"})
             .sort_values("avail"),
            left_on="idx", right_on="avail", direction="backward",
        ).set_index("idx")
        out["holders_date"] = merged["holders_date_"].reindex(out.index)
        out["big_pct"] = merged["big_pct"].reindex(out.index)
        out["big_chg_1w"] = merged["c1"].reindex(out.index)
        out["big_chg_4w"] = merged["c4"].reindex(out.index)

    # 籌碼訊號 A：延伸過度 + 大戶單週大增 → 別追（可能是一次性轉倉或末升段追價）
    out["chips_surge_ext"] = out["is_extended"] & (out["big_chg_1w"] >= surge_pts)
    # 籌碼訊號 B：大戶 4 週累計減持 → 派發（Stage 3 的籌碼面佐證）
    out["chips_distribution"] = out["big_chg_4w"] <= -dist_pts

    had_recent_exhaustion = out["exhaustion_gap"].rolling(60, min_periods=1).max().astype(bool)
    slope_decelerating = out["ma30w_slope"] < out["ma30w_slope"].shift(20)
    price_below_ma = out["close"] < out["ma30w"]
    stage3_price = had_recent_exhaustion & slope_decelerating & price_below_ma

    out["stage3_chips_confirmed"] = stage3_price & out["chips_distribution"]
    out["stage3_top_signal"] = out["stage3_chips_confirmed"] if require_chips else stage3_price

    return out


def weekly_chip_view(out: pd.DataFrame) -> pd.DataFrame:
    """把日線上的籌碼欄位彙整回每週一列，供副圖與表格使用。"""
    sub = out.dropna(subset=["holders_date"])
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("holders_date").agg(
        big_pct=("big_pct", "first"),
        chg_1w=("big_chg_1w", "first"),
        chg_4w=("big_chg_4w", "first"),
        surge_ext=("chips_surge_ext", "max"),
        distribution=("chips_distribution", "max"),
    )
    g.index.name = "date"
    return g


def plot_chart(out: pd.DataFrame, ticker: str):
    weekly = weekly_chip_view(out)
    has_chips = not weekly.empty

    ratios = [3, 1, 1.3] if has_chips else [3, 1]
    fig, axes = plt.subplots(
        len(ratios), 1, figsize=(13, 10 if has_chips else 8), sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(out.index, out["close"], color="black", lw=1.1, label="收盤價")
    ax1.plot(out.index, out["ma30w"], color="tab:blue", lw=1.4, label="30週均線 (MA30W)")

    ext = out[out["is_extended"]]
    ax1.scatter(ext.index, ext["close"], s=10, color="orange", alpha=0.5, label="延伸過度（禁止追高區）")

    gaps = out[out["exhaustion_gap"]]
    ax1.scatter(gaps.index, gaps["close"], marker="^", s=140, color="red",
                edgecolor="black", zorder=5, label="耗盡缺口 (Exhaustion Gap)")

    if has_chips:
        surge_days = out[out["chips_surge_ext"]].drop_duplicates("holders_date")
        ax1.scatter(surge_days.index, surge_days["close"], marker="D", s=70, color="magenta",
                    edgecolor="black", zorder=6, label="延伸 + 大戶單週大增（別追）")

    tops = out[out["stage3_top_signal"]]
    if not tops.empty:
        ax1.axvspan(tops.index[0], out.index[-1], color="red", alpha=0.08, label="Stage 3 出場區")

    ax1.set_title(f"{ticker} — Weinstein 階段分析", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.bar(out.index, out["volume"], color=np.where(out["volume_spike"], "red", "gray"), width=1.0)
    ax2.set_ylabel("成交量")
    ax2.grid(alpha=0.25)

    if has_chips:
        ax3 = axes[2]
        ax3.step(weekly.index, weekly["big_pct"], where="post", color="tab:red", lw=1.5, label="大戶持股 (%)")
        ax3.scatter(weekly.index, weekly["big_pct"], s=14, color="tab:red", zorder=3)
        surge_w = weekly[weekly["surge_ext"].astype(bool)]
        ax3.scatter(surge_w.index, surge_w["big_pct"], marker="D", s=70, color="magenta",
                    edgecolor="black", zorder=5, label="延伸 + 大增")
        dist_w = weekly[weekly["distribution"].astype(bool)]
        ax3.scatter(dist_w.index, dist_w["big_pct"], marker="v", s=70, color="green",
                    edgecolor="black", zorder=5, label="4 週減持（派發）")
        ax3.set_ylabel("大戶持股 (%)")
        ax3.grid(alpha=0.25)

        ax3b = ax3.twinx()
        chg = weekly["chg_1w"].fillna(0)
        ax3b.bar(weekly.index, chg, width=4,
                 color=np.where(chg >= 0, "tab:red", "tab:green"), alpha=0.35)
        ax3b.axhline(0, color="gray", lw=0.6)
        ax3b.set_ylabel("週增減 (點)")
        ax3.legend(loc="upper left", fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Streamlit 介面
# ----------------------------------------------------------------------
st.set_page_config(page_title="Weinstein 階段分析", layout="wide")
st.title("📈 Stan Weinstein 階段分析：耗盡缺口 + Stage 3 出場訊號")
st.caption("永遠不要買入延伸 — 找出過度延伸、耗盡缺口與 Stage 3 頂部退出時機；台股另加大戶持股籌碼")

with st.sidebar:
    st.header("查詢設定")
    ticker = st.text_input(
        "個股代號",
        value="AAPL",
        help="美股直接輸入代號（如 AAPL、TSLA）；台股請加 .TW（如 2330.TW）；加 .TWO 為上櫃",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input(
            "起始日期", value=dt.date.today() - dt.timedelta(days=365 * 3)
        )
    with col_b:
        end_date = st.date_input("結束日期", value=dt.date.today())

    st.divider()
    st.subheader("分析參數")
    ma_window = st.slider("均線週期（交易日，30週≈150日）", 50, 250, 150, step=10)
    extension_threshold = st.slider("延伸過度門檻（乖離率）", 0.10, 1.00, 0.30, step=0.05)
    gap_atr_mult = st.slider("耗盡缺口跳空門檻（ATR 倍數）", 0.5, 4.0, 2.0, step=0.1)
    volume_mult = st.slider(
        "爆量門檻（成交量 ÷ 20日均量）", 1.0, 4.0, 1.7, step=0.1,
        help="調低會更容易判定為爆量；1.0 等於不要求爆量",
    )
    volume_days = st.slider(
        "爆量容許天數（缺口當日起算）", 1, 10, 3, step=1,
        help="跳空當天起算，往後幾個交易日內出現爆量即算成立",
    )

    st.divider()
    st.subheader("籌碼：大戶持股（僅台股）")
    tw = is_tw_ticker(ticker)
    if not tw:
        st.caption("目前代號不是台股（.TW / .TWO），籌碼功能停用。")
    use_chips = st.checkbox("載入大戶持股", value=True, disabled=not tw)
    big_thr = st.select_slider("大戶定義（持股張數以上）", options=list(THRESHOLDS), value=1000,
                               disabled=not tw)
    surge_pts = st.slider(
        "大戶單週大增門檻（百分點）", 1.0, 10.0, 3.0, step=0.5, disabled=not tw,
        help="延伸過度時，大戶持股單週增加超過此值 → 標示「別追」",
    )
    dist_pts = st.slider(
        "大戶 4 週減持門檻（百分點）", 0.5, 10.0, 1.0, step=0.5, disabled=not tw,
        help="大戶持股 4 週累計減少超過此值 → 視為派發",
    )
    require_chips = st.checkbox(
        "Stage 3 需大戶減持確認", value=False, disabled=not tw,
        help="勾選後，Stage 3 出場訊號必須同時有大戶 4 週減持才成立；不勾則只作為佐證顯示",
    )
    finmind_token = st.text_input(
        "FinMind Token（選填）", type="password", disabled=not tw,
        help="股權分散表歷史資料需 FinMind backer/sponsor 會員。沒有 token 時只會用本機快取（先前用 token 取得的資料）+ 手動貼上的資料。",
    )
    manual_text = st.text_area(
        "手動貼上大戶持股（選填）", height=140, disabled=not tw,
        placeholder="每行：日期 大戶持股%\n2026-10-16 63.40\n10/08 52.91\n10/02 53.08",
        help="可直接從籌碼 K 線 app 的表格複製；每行日期後的第一個數字視為大戶持股%。手動資料會覆蓋同日期的其他來源。",
    )

    run_btn = st.button("開始分析", type="primary", use_container_width=True)

if run_btn:
    if start_date >= end_date:
        st.error("起始日期必須早於結束日期。")
        st.stop()

    with st.spinner(f"下載 {ticker} 資料中..."):
        df = load_price_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"找不到 {ticker} 的資料，請確認代號是否正確（台股記得加 .TW / .TWO）。")
        st.stop()

    holders = None
    if tw and use_chips:
        with st.spinner("載入大戶持股..."):
            holders, notes = build_holders_series(
                stock_id_of(ticker), big_thr, start_date, end_date,
                token=finmind_token, manual_text=manual_text,
            )
        for n in notes:
            st.caption(f"籌碼資料：{n}")
        if holders.empty:
            st.warning("沒有取得任何大戶持股資料，籌碼副圖與訊號已略過。請在左側手動貼上資料，或填入 FinMind token。")
            holders = None
        elif len(holders) < 6:
            st.info(
                f"目前只有 {len(holders)} 週大戶持股資料，副圖與「4 週減持」判斷會很有限。"
                "可手動貼上更多週的資料，或提供 FinMind token 取得完整歷史。"
            )

    effective_require = require_chips and holders is not None
    if require_chips and holders is None:
        st.warning("沒有大戶持股資料，已忽略「Stage 3 需大戶減持確認」。")

    result = weinstein_analysis(
        df,
        ma_window=ma_window,
        extension_threshold=extension_threshold,
        gap_atr_mult=gap_atr_mult,
        volume_mult=volume_mult,
        volume_days=volume_days,
        holders=holders,
        surge_pts=surge_pts,
        dist_pts=dist_pts,
        require_chips=effective_require,
    )

    latest = result.iloc[-1]
    cols = st.columns(5 if holders is not None else 4)
    cols[0].metric("最新收盤", f"{latest['close']:.2f}")
    cols[1].metric("乖離率 (vs MA30W)", f"{latest['deviation_pct']*100:.1f}%")
    cols[2].metric("目前延伸過度？", "是 ⚠️" if latest["is_extended"] else "否")
    cols[3].metric("目前 Stage 3 訊號？", "是 🔴" if latest["stage3_top_signal"] else "否")
    if holders is not None and pd.notna(latest["big_pct"]):
        chg = latest["big_chg_1w"]
        cols[4].metric(
            f"大戶持股（≥{big_thr}張）",
            f"{latest['big_pct']:.2f}%",
            None if pd.isna(chg) else f"{chg:+.2f} 點（週）",
        )

    if holders is not None:
        if latest["chips_surge_ext"]:
            st.warning("⚠️ 目前處於延伸過度，且大戶單週大增：不要追價，先確認是否為一次性轉倉。")
        if latest["chips_distribution"] and latest["is_extended"]:
            st.warning("⚠️ 延伸過度且大戶 4 週累計減持：留意派發。")

    st.pyplot(plot_chart(result, ticker))

    st.subheader("耗盡缺口紀錄")
    gaps = result[result["exhaustion_gap"]][["close", "deviation_pct", "gap", "volume"]]
    if gaps.empty:
        st.info("此區間內未偵測到耗盡缺口。可嘗試調低「跳空門檻」或「爆量門檻」。")
    else:
        gaps_display = gaps.copy()
        gaps_display.index = gaps_display.index.strftime("%Y-%m-%d")
        gaps_display.index.name = "日期"
        gaps_display["deviation_pct"] = (gaps_display["deviation_pct"] * 100).round(1).astype(str) + "%"
        gaps_display.columns = ["收盤價", "乖離率", "跳空幅度", "成交量"]
        st.dataframe(gaps_display, use_container_width=True)

    st.subheader("Stage 3 出場區間")
    tops = result[result["stage3_top_signal"]]
    if tops.empty:
        st.info("此區間內尚未觸發 Stage 3 出場訊號。")
    else:
        st.warning(
            f"首次觸發日：{tops.index[0].date()}，收盤 {tops.iloc[0]['close']:.2f}"
            f"（此後共 {len(tops)} 個交易日持續處於出場區）"
        )
        if holders is not None:
            confirmed = int(result["stage3_chips_confirmed"].sum())
            st.caption(f"其中有大戶減持佐證的交易日：{confirmed} 天。")

    if holders is not None:
        st.subheader("大戶持股週紀錄")
        wk = weekly_chip_view(result)
        if wk.empty:
            st.info("所選日期區間內沒有可對應的大戶持股資料。")
        else:
            show = wk.sort_index(ascending=False).copy()
            show.index = show.index.strftime("%Y-%m-%d")
            show.index.name = "日期"
            show["surge_ext"] = show["surge_ext"].map({True: "⚠️", False: ""})
            show["distribution"] = show["distribution"].map({True: "🔻", False: ""})
            show.columns = [f"大戶持股≥{big_thr}張(%)", "週增減(點)", "4週增減(點)", "延伸+大增", "4週減持"]
            st.dataframe(show.round(2), use_container_width=True)
else:
    st.info("請在左側輸入個股代號與日期範圍，然後按「開始分析」。")
