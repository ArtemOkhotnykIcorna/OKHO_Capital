#!/usr/bin/env python3
"""
BTC+ETH Futures Bot (more stable swing-ish)
- MAINNET futures data for signals (clean candles)
- TESTNET futures for execution

Stability:
- HTF(15m): trend + ATR
- LTF(5m): entry timing + exit checks
Risk:
- SL = entry +/- k_sl * ATR(HTF)
- TP1: close 50% at RR=1.0
- After TP1: move SL to breakeven (entry) and trail remainder
- TP2: RR=2.5 target (or trailing stop closes earlier)

WARNING: For testing mechanics only.
"""

import time
import csv
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

import requests
import numpy as np
import pandas as pd

from binance.client import Client
from binance.exceptions import BinanceAPIException


# =========================
# 0) USER SETTINGS
# =========================

API_KEY = "Z2WxlFKF7Vo2jnNmQf5yOZfiyYsemhkIz6QsjnRv4YCDmamaQVJEh7SGcTmqNBw3"
API_SECRET = "ahdaf3j3Tu76WOJAqHLYJ8TUjAymRFcwz3Y9NR2SClE2xYjlDgzxtZ1LfR57xssf"
USE_TESTNET = True

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

HTF_INTERVAL = "15m"
LTF_INTERVAL = "5m"

LIMIT_HTF = 600
LIMIT_LTF = 600

LEVERAGE = 10
USD_POSITION_SIZE = 50.0   # USDT per trade per symbol

POLL_INTERVAL_SEC = 15

# MAINNET futures data source
FUTURES_DATA_BASE_URL = "https://fapi.binance.com"


# =========================
# 1) Client / Exchange helpers
# =========================

def create_client() -> Client:
    c = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        c.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return c


client = create_client()
SYMBOL_FILTERS: Dict[str, Dict[str, float]] = {}


def load_symbol_filters():
    """Load lot size filters (stepSize/minQty) from TESTNET exchangeInfo."""
    global SYMBOL_FILTERS
    try:
        info = client.futures_exchange_info()
    except BinanceAPIException as e:
        print(f"[ERROR] futures_exchange_info failed: {e}")
        return

    for s in info.get("symbols", []):
        symbol = s.get("symbol")
        lot_size = next((f for f in s.get("filters", [])
                         if f.get("filterType") in ("LOT_SIZE", "MARKET_LOT_SIZE")), None)
        if symbol and lot_size:
            SYMBOL_FILTERS[symbol] = {
                "stepSize": float(lot_size.get("stepSize", "0")),
                "minQty": float(lot_size.get("minQty", "0")),
            }


def round_qty(symbol: str, qty: float) -> float:
    """Floor qty to stepSize."""
    f = SYMBOL_FILTERS.get(symbol)
    if not f:
        return float(f"{qty:.6f}")

    step = f["stepSize"]
    min_qty = f["minQty"]

    if step <= 0:
        q = float(f"{qty:.6f}")
        return q if q >= min_qty else 0.0

    floored = (qty // step) * step
    floored = float(f"{floored:.8f}")
    return floored if floored >= min_qty else 0.0


def ensure_leverage(symbol: str, leverage: int):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"[CONFIG] Leverage {leverage}x set for {symbol}")
    except BinanceAPIException as e:
        print(f"[WARN] Cannot set leverage for {symbol}: {e}")


def open_position_market(symbol: str, direction: str, price_hint: float) -> Tuple[Optional[float], Optional[float]]:
    """Open MARKET position on TESTNET. Return (qty, fill_price approx)."""
    side = "BUY" if direction == "LONG" else "SELL"
    qty = USD_POSITION_SIZE / max(price_hint, 1e-9)
    qty = round_qty(symbol, qty)
    if qty <= 0:
        print(f"[ERROR] Qty too small for {symbol}: {qty}")
        return None, None

    try:
        order = client.futures_create_order(
            symbol=symbol,
            type="MARKET",
            side=side,
            quantity=qty,
        )
    except BinanceAPIException as e:
        print(f"[ERROR] open_position_market failed: {e}")
        return None, None

    fill = None
    try:
        if "avgPrice" in order and float(order["avgPrice"]) > 0:
            fill = float(order["avgPrice"])
    except Exception:
        fill = None

    if fill is None:
        fill = float(price_hint)

    print(f"[OPEN] {direction} {symbol} qty={qty} avg≈{fill:.6f}")
    return qty, fill


def close_position_market(symbol: str, direction: str, qty: float):
    """Close by reduceOnly MARKET."""
    qty = round_qty(symbol, qty)
    if qty <= 0:
        return
    side = "SELL" if direction == "LONG" else "BUY"
    try:
        client.futures_create_order(
            symbol=symbol,
            type="MARKET",
            side=side,
            quantity=qty,
            reduceOnly=True,
        )
        print(f"[CLOSE] {direction} {symbol} qty={qty}")
    except BinanceAPIException as e:
        print(f"[ERROR] close_position_market failed: {e}")


def close_partial_market(symbol: str, direction: str, qty_to_close: float) -> float:
    """
    Close part of position with reduceOnly MARKET.
    Returns actual qty used after rounding (0.0 if too small).
    """
    qty_to_close = round_qty(symbol, qty_to_close)
    if qty_to_close <= 0:
        return 0.0
    side = "SELL" if direction == "LONG" else "BUY"
    try:
        client.futures_create_order(
            symbol=symbol,
            type="MARKET",
            side=side,
            quantity=qty_to_close,
            reduceOnly=True,
        )
        print(f"[TP1] Partial close {direction} {symbol} qty={qty_to_close}")
        return qty_to_close
    except BinanceAPIException as e:
        print(f"[ERROR] close_partial_market failed: {e}")
        return 0.0


# =========================
# 2) Config & Position
# =========================

Signal = Literal["LONG", "SHORT", "WAIT"]


@dataclass
class BotConfig:
    # Features
    ema_fast: int = 9
    ema_slow: int = 26
    atr_window: int = 14
    adx_period: int = 10
    vol_ma_window: int = 15
    z_window: int = 40
    vol_window: int = 32
    mom_lookback: int = 16

    # HTF trend filters (stricter)
    ema_spread_threshold: float = 0.0012  # 0.12%
    min_adx: float = 18.0
    max_adx: float = 45.0

    # LTF filters
    min_volume_ratio: float = 0.25
    min_realized_vol: float = 0.0010
    max_realized_vol: float = 0.0065
    enable_confirmation_candle: bool = True
    body_ratio_max: float = 0.85

    # Risk model from HTF ATR
    k_sl: float = 1.8

    # Partial TP scheme
    tp1_rr: float = 1.0      # TP1 at RR=1
    tp1_close_frac: float = 0.50  # close 50%
    tp2_rr: float = 2.5      # final target RR=2.5

    # After TP1 behavior
    move_sl_to_be: bool = True    # move SL to breakeven after TP1

    # Trailing for the remainder
    enable_trailing: bool = True
    trailing_atr_mult: float = 0.9

    # Max holding time on LTF (5m)
    max_holding_bars_ltf: int = 144  # 144 * 5m = 12h

    poll_interval_sec: int = POLL_INTERVAL_SEC

    actions_log: str = "actions_log.csv"
    trades_log: str = "trades_log.csv"


@dataclass
class Position:
    is_open: bool
    direction: Optional[Literal["LONG", "SHORT"]] = None

    entry_price: Optional[float] = None
    qty_total: float = 0.0
    qty_remaining: float = 0.0

    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None

    atr_htf: Optional[float] = None
    risk_abs: Optional[float] = None  # abs distance entry->SL

    tp1_done: bool = False

    bars_in_trade_ltf: int = 0
    entry_time: Optional[pd.Timestamp] = None
    last_update_time: Optional[pd.Timestamp] = None
    entry_reason: Optional[str] = None


# =========================
# 3) Data (MAINNET futures klines)
# =========================

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{FUTURES_DATA_BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("close_time", inplace=True)
    return df


# =========================
# 4) Features
# =========================

def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def compute_features(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    df = df.copy()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv"] = df["log_ret"].rolling(cfg.vol_window).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)

    df["momentum"] = df["close"] / df["close"].shift(cfg.mom_lookback) - 1.0

    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    close_prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close_prev).abs()
    tr3 = (df["low"] - close_prev).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(cfg.atr_window).mean()

    df["adx"] = compute_adx(df, cfg.adx_period)

    df["vol_ma"] = df["volume"].rolling(cfg.vol_ma_window).mean()
    df["sma"] = df["close"].rolling(cfg.z_window).mean()
    df["sma_std"] = df["close"].rolling(cfg.z_window).std()
    df["z_score"] = (df["close"] - df["sma"]) / df["sma_std"]
    return df


# =========================
# 5) Filters / Signals
# =========================

def is_spiky_candle(latest: pd.Series, atr: float) -> bool:
    high = float(latest["high"])
    low = float(latest["low"])
    open_ = float(latest["open"])
    close = float(latest["close"])

    rng = high - low
    body = abs(close - open_)

    if np.isnan(atr) or atr <= 0:
        return False

    if rng > 2.5 * atr:
        return True

    if body > 0:
        upper = high - max(open_, close)
        lower = min(open_, close) - low
        if upper > 3 * body or lower > 3 * body:
            return True

    return False


def decide_trend_from_htf(htf_last: pd.Series, cfg: BotConfig) -> Tuple[Signal, Dict[str, Any]]:
    price = float(htf_last["close"])
    ema_fast = float(htf_last["ema_fast"])
    ema_slow = float(htf_last["ema_slow"])
    adx = float(htf_last.get("adx", np.nan))
    atr = float(htf_last.get("atr", np.nan))

    info = {
        "price": price,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "adx": adx,
        "atr": atr,
    }

    if price <= 0 or np.isnan(ema_fast) or np.isnan(ema_slow):
        return "WAIT", {**info, "reason": "htf_no_ema"}

    spread = (ema_fast - ema_slow) / price
    info["ema_spread"] = spread

    if np.isnan(adx) or adx < cfg.min_adx or adx > cfg.max_adx:
        return "WAIT", {**info, "reason": "htf_adx_out_of_range"}

    if spread > cfg.ema_spread_threshold:
        return "LONG", {**info, "reason": "htf_uptrend"}
    if spread < -cfg.ema_spread_threshold:
        return "SHORT", {**info, "reason": "htf_downtrend"}

    return "WAIT", {**info, "reason": "htf_no_clear_trend"}


def decide_entry_from_ltf(ltf_last: pd.Series, allowed: Signal, cfg: BotConfig) -> Tuple[Signal, Dict[str, Any]]:
    price = float(ltf_last["close"])
    open_ = float(ltf_last["open"])
    close = float(ltf_last["close"])
    high = float(ltf_last["high"])
    low = float(ltf_last["low"])

    atr = float(ltf_last.get("atr", np.nan))
    rv = float(ltf_last.get("rv", np.nan))
    vol = float(ltf_last.get("volume", np.nan))
    vol_ma = float(ltf_last.get("vol_ma", np.nan))

    info = {
        "price": price,
        "atr": atr,
        "rv": rv,
        "volume": vol,
        "vol_ma": vol_ma,
        "allowed": allowed,
    }

    if allowed not in ("LONG", "SHORT"):
        return "WAIT", {**info, "reason": "htf_blocks_entry"}

    if is_spiky_candle(ltf_last, atr):
        return "WAIT", {**info, "reason": "ltf_spiky_filtered"}

    if not np.isnan(vol_ma) and vol_ma > 0 and not np.isnan(vol):
        if vol < cfg.min_volume_ratio * vol_ma:
            return "WAIT", {**info, "reason": "ltf_low_volume"}

    if np.isnan(rv) or rv < cfg.min_realized_vol:
        return "WAIT", {**info, "reason": "ltf_low_volatility"}
    if cfg.max_realized_vol is not None and rv > cfg.max_realized_vol:
        return "WAIT", {**info, "reason": "ltf_too_much_volatility"}

    full_range = max(high - low, 1e-9)
    body = abs(close - open_)
    body_ratio = body / full_range
    info["body_ratio"] = body_ratio
    if body_ratio > cfg.body_ratio_max:
        return "WAIT", {**info, "reason": "ltf_explosive_candle"}

    if cfg.enable_confirmation_candle:
        if allowed == "LONG" and close <= open_:
            return "WAIT", {**info, "reason": "ltf_no_bullish_close"}
        if allowed == "SHORT" and close >= open_:
            return "WAIT", {**info, "reason": "ltf_no_bearish_close"}

    return allowed, {**info, "reason": "ltf_entry_ok"}


def compute_levels_partial(direction: str, entry: float, atr_htf: float, cfg: BotConfig) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    SL = entry +/- k_sl * ATR(HTF)
    TP1 = entry +/- RR1 * risk
    TP2 = entry +/- RR2 * risk
    """
    if np.isnan(atr_htf) or atr_htf <= 0:
        atr_htf = entry * 0.003  # fallback ~0.3%

    risk = cfg.k_sl * atr_htf

    if direction == "LONG":
        sl = entry - risk
        tp1 = entry + cfg.tp1_rr * risk
        tp2 = entry + cfg.tp2_rr * risk
    elif direction == "SHORT":
        sl = entry + risk
        tp1 = entry - cfg.tp1_rr * risk
        tp2 = entry - cfg.tp2_rr * risk
    else:
        raise ValueError("direction must be LONG/SHORT")

    meta = {
        "atr_htf_used": atr_htf,
        "k_sl": cfg.k_sl,
        "risk_abs": risk,
        "tp1_rr": cfg.tp1_rr,
        "tp2_rr": cfg.tp2_rr,
    }
    return sl, tp1, tp2, risk, meta


# =========================
# 6) Exit logic (LTF checks)
# =========================

def update_position_with_ltf_bar(pos: Position, ltf_last: pd.Series, cfg: BotConfig) -> Tuple[Position, Optional[Dict[str, Any]]]:
    """
    Returns exit_event:
    - TP1 event (partial) -> caller executes partial close and updates pos
    - FULL CLOSE event -> caller closes remainder and logs trade
    - None -> hold
    """
    if not pos.is_open:
        return pos, None

    high = float(ltf_last["high"])
    low = float(ltf_last["low"])
    close = float(ltf_last["close"])
    atr_ltf = float(ltf_last.get("atr", np.nan))
    now = ltf_last.name

    event = None

    # 1) STOP LOSS (always full close)
    if pos.direction == "LONG":
        if low <= pos.sl:
            event = {"type": "CLOSE_SL", "reason": "hit_sl", "price": float(pos.sl), "bar_time": now.isoformat()}
            return _finalize_close(pos, now), event
    else:
        if high >= pos.sl:
            event = {"type": "CLOSE_SL", "reason": "hit_sl", "price": float(pos.sl), "bar_time": now.isoformat()}
            return _finalize_close(pos, now), event

    # 2) TP1 (partial) if not done
    if not pos.tp1_done:
        if pos.direction == "LONG" and high >= pos.tp1:
            event = {"type": "TP1_PARTIAL", "reason": "hit_tp1", "price": float(pos.tp1), "bar_time": now.isoformat()}
            pos.last_update_time = now
            return pos, event
        if pos.direction == "SHORT" and low <= pos.tp1:
            event = {"type": "TP1_PARTIAL", "reason": "hit_tp1", "price": float(pos.tp1), "bar_time": now.isoformat()}
            pos.last_update_time = now
            return pos, event

    # 3) TP2 (final) if hit (full close remainder)
    if pos.direction == "LONG":
        if high >= pos.tp2:
            event = {"type": "CLOSE_TP2", "reason": "hit_tp2", "price": float(pos.tp2), "bar_time": now.isoformat()}
            return _finalize_close(pos, now), event
    else:
        if low <= pos.tp2:
            event = {"type": "CLOSE_TP2", "reason": "hit_tp2", "price": float(pos.tp2), "bar_time": now.isoformat()}
            return _finalize_close(pos, now), event

    # 4) time stop
    pos.bars_in_trade_ltf += 1
    if pos.bars_in_trade_ltf >= cfg.max_holding_bars_ltf:
        event = {"type": "CLOSE_TIME", "reason": "max_holding_bars_ltf", "price": float(close), "bar_time": now.isoformat()}
        return _finalize_close(pos, now), event

    # 5) trailing (only meaningful after TP1, but you can allow always; we’ll bias after TP1)
    if cfg.enable_trailing:
        atr_ref = pos.atr_htf
        if atr_ref is None or np.isnan(atr_ref) or atr_ref <= 0:
            atr_ref = atr_ltf

        if atr_ref is not None and not np.isnan(atr_ref) and atr_ref > 0:
            # After TP1 we trail more confidently; before TP1 we keep SL as-is.
            if pos.tp1_done:
                if pos.direction == "LONG":
                    unreal = close - pos.entry_price
                    if unreal > cfg.trailing_atr_mult * atr_ref:
                        new_sl = close - cfg.trailing_atr_mult * atr_ref
                        pos.sl = max(pos.sl, new_sl)
                else:
                    unreal = pos.entry_price - close
                    if unreal > cfg.trailing_atr_mult * atr_ref:
                        new_sl = close + cfg.trailing_atr_mult * atr_ref
                        pos.sl = min(pos.sl, new_sl)

    pos.last_update_time = now
    return pos, None


def _finalize_close(pos: Position, now: pd.Timestamp) -> Position:
    pos.is_open = False
    pos.last_update_time = now
    return pos


# =========================
# 7) Logging
# =========================

def ensure_csv_headers(filepath: str, fieldnames):
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def log_action(cfg: BotConfig, info: Dict[str, Any]):
    fields = sorted(info.keys())
    ensure_csv_headers(cfg.actions_log, fields)
    with open(cfg.actions_log, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(info)


def log_trade(cfg: BotConfig, pos: Position, exit_info: Dict[str, Any], pnl_abs: float, pnl_pct: float):
    row = {
        "symbol": exit_info.get("symbol", ""),
        "entry_time": pos.entry_time.isoformat() if pos.entry_time else "",
        "exit_time": exit_info.get("exit_bar_time", exit_info.get("bar_time", "")),
        "direction": pos.direction,
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_info.get("exit_price", pos.entry_price)),
        "pnl_abs": float(pnl_abs),
        "pnl_pct": float(pnl_pct),
        "reason": exit_info.get("exit_reason", ""),
        "size": float(exit_info.get("size", pos.qty_total)),
    }

    fields = ["symbol","entry_time","exit_time","direction","entry_price","exit_price","pnl_abs","pnl_pct","reason","size"]
    ensure_csv_headers(cfg.trades_log, fields)
    with open(cfg.trades_log, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# =========================
# 8) PnL helpers (approx, without fees)
# =========================

def pnl_for_close(direction: str, entry: float, exit_price: float, qty: float) -> Tuple[float, float]:
    if qty <= 0:
        return 0.0, 0.0
    if direction == "LONG":
        pnl_abs = (exit_price - entry) * qty
        pnl_pct = (exit_price - entry) / entry
    else:
        pnl_abs = (entry - exit_price) * qty
        pnl_pct = (entry - exit_price) / entry
    return pnl_abs, pnl_pct


# =========================
# 9) Main loop
# =========================

def main():
    cfg = BotConfig()

    print("=== BTC+ETH Partial TP Bot (MAINNET data, TESTNET execution) ===")
    print("HTF:", HTF_INTERVAL, "LTF:", LTF_INTERVAL)
    print(f"TP1: close {int(cfg.tp1_close_frac*100)}% at RR={cfg.tp1_rr} | TP2: RR={cfg.tp2_rr} + trailing after TP1")
    print("USD_POSITION_SIZE:", USD_POSITION_SIZE, "LEVERAGE:", LEVERAGE)
    print()

    load_symbol_filters()
    for s in SYMBOLS:
        ensure_leverage(s, LEVERAGE)

    positions: Dict[str, Position] = {s: Position(is_open=False) for s in SYMBOLS}
    last_ltf_bar_time: Dict[str, Optional[pd.Timestamp]] = {s: None for s in SYMBOLS}

    while True:
        try:
            for symbol in SYMBOLS:
                df_htf = compute_features(fetch_klines(symbol, HTF_INTERVAL, LIMIT_HTF), cfg)
                df_ltf = compute_features(fetch_klines(symbol, LTF_INTERVAL, LIMIT_LTF), cfg)

                htf_last = df_htf.iloc[-1]
                ltf_last = df_ltf.iloc[-1]
                ltf_time = df_ltf.index[-1]
                atr_htf = float(htf_last.get("atr", np.nan))

                allowed, htf_diag = decide_trend_from_htf(htf_last, cfg)

                # New LTF bar?
                if last_ltf_bar_time[symbol] is None or ltf_time > last_ltf_bar_time[symbol]:
                    last_ltf_bar_time[symbol] = ltf_time

                    pos = positions[symbol]
                    was_open = pos.is_open

                    base_info: Dict[str, Any] = {
                        "bar_time": ltf_time.isoformat(),
                        "symbol": symbol,
                        "price": float(ltf_last["close"]),
                        "position_was_open": was_open,
                        "allowed_dir": allowed,
                        "htf_reason": htf_diag.get("reason"),
                        "htf_adx": htf_diag.get("adx"),
                        "htf_ema_spread": htf_diag.get("ema_spread"),
                        "htf_atr": htf_diag.get("atr"),
                    }

                    # --- Manage open position ---
                    if pos.is_open:
                        positions[symbol], event = update_position_with_ltf_bar(pos, ltf_last, cfg)
                        pos2 = positions[symbol]

                        if event is None:
                            info = {**base_info, "action": "HOLD", "sl": pos2.sl, "tp1": pos2.tp1, "tp2": pos2.tp2, "tp1_done": pos2.tp1_done}
                            print(f"[BAR] {symbol} {info['bar_time']} action=HOLD price={info['price']}")
                            log_action(cfg, info)
                            continue

                        # --- TP1 partial ---
                        if event["type"] == "TP1_PARTIAL":
                            qty_to_close = pos2.qty_remaining * cfg.tp1_close_frac
                            qty_closed = close_partial_market(symbol, pos2.direction, qty_to_close)

                            # If we could close something:
                            if qty_closed > 0:
                                # Bookkeeping
                                pos2.qty_remaining = max(0.0, pos2.qty_remaining - qty_closed)
                                pos2.tp1_done = True

                                # Move SL to breakeven (entry) after TP1
                                if cfg.move_sl_to_be and pos2.entry_price is not None:
                                    pos2.sl = float(pos2.entry_price)

                                # Log TP1 as an action (and also as a trade row for closed part)
                                pnl_abs, pnl_pct = pnl_for_close(pos2.direction, pos2.entry_price, event["price"], qty_closed)

                                info = {
                                    **base_info,
                                    "action": "TP1_PARTIAL",
                                    "exit_reason": "tp1_partial_50pct",
                                    "exit_price": event["price"],
                                    "closed_qty": qty_closed,
                                    "remaining_qty": pos2.qty_remaining,
                                    "new_sl": pos2.sl,
                                    "tp2": pos2.tp2,
                                }
                                print(f"[BAR] {symbol} {info['bar_time']} action=TP1_PARTIAL exit={info['exit_price']} closed_qty={qty_closed}")
                                log_action(cfg, info)

                                trade_info = {
                                    "symbol": symbol,
                                    "exit_reason": "TP1_PARTIAL",
                                    "exit_price": event["price"],
                                    "exit_bar_time": event["bar_time"],
                                    "size": qty_closed,
                                }
                                log_trade(cfg, pos2, trade_info, pnl_abs, pnl_pct)

                            else:
                                info = {**base_info, "action": "TP1_FAILED_TOO_SMALL_OR_ERROR"}
                                print(f"[BAR] {symbol} {info['bar_time']} action=TP1_FAILED")
                                log_action(cfg, info)

                            positions[symbol] = pos2
                            continue

                        # --- Final close (SL/TP2/TIME) ---
                        final_reason = event["type"]
                        exit_price = event["price"]

                        # close remaining qty on exchange
                        if pos2.direction and pos2.qty_remaining > 0:
                            close_position_market(symbol, pos2.direction, pos2.qty_remaining)

                        # compute pnl for remaining qty only (since TP1 already logged)
                        qty_rem = pos2.qty_remaining
                        pnl_abs, pnl_pct = pnl_for_close(pos2.direction, pos2.entry_price, exit_price, qty_rem)

                        info = {
                            **base_info,
                            "action": final_reason,
                            "exit_reason": event["reason"],
                            "exit_price": exit_price,
                            "closed_qty": qty_rem,
                            "tp1_done": pos2.tp1_done,
                        }
                        print(f"[BAR] {symbol} {info['bar_time']} action={final_reason} exit={exit_price}")
                        log_action(cfg, info)

                        trade_info = {
                            "symbol": symbol,
                            "exit_reason": final_reason,
                            "exit_price": exit_price,
                            "exit_bar_time": event["bar_time"],
                            "size": qty_rem,
                        }
                        log_trade(cfg, pos2, trade_info, pnl_abs, pnl_pct)

                        # reset position object
                        positions[symbol] = Position(is_open=False)
                        continue

                    # --- No position: decide entry ---
                    entry_signal, ltf_diag = decide_entry_from_ltf(ltf_last, allowed, cfg)

                    if entry_signal in ("LONG", "SHORT"):
                        price_hint = float(ltf_last["close"])
                        qty_total, fill_price = open_position_market(symbol, entry_signal, price_hint)

                        if qty_total is None:
                            info = {**base_info, "action": "OPEN_FAILED"}
                            print(f"[BAR] {symbol} {info['bar_time']} action=OPEN_FAILED")
                            log_action(cfg, info)
                            continue

                        sl, tp1, tp2, risk_abs, meta = compute_levels_partial(entry_signal, fill_price, atr_htf, cfg)

                        positions[symbol] = Position(
                            is_open=True,
                            direction=entry_signal,
                            entry_price=fill_price,
                            qty_total=qty_total,
                            qty_remaining=qty_total,
                            sl=sl,
                            tp1=tp1,
                            tp2=tp2,
                            atr_htf=meta["atr_htf_used"],
                            risk_abs=risk_abs,
                            tp1_done=False,
                            bars_in_trade_ltf=0,
                            entry_time=ltf_time,
                            last_update_time=ltf_time,
                            entry_reason=ltf_diag.get("reason"),
                        )

                        info = {
                            **base_info,
                            "action": f"OPEN_{entry_signal}",
                            "entry_price": fill_price,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "risk_abs": risk_abs,
                            "tp1_rr": cfg.tp1_rr,
                            "tp2_rr": cfg.tp2_rr,
                            "qty_total": qty_total,
                        }
                        print(f"[BAR] {symbol} {info['bar_time']} action={info['action']} entry={fill_price}")
                        log_action(cfg, info)
                    else:
                        info = {**base_info, "action": "NO_TRADE", "reason": ltf_diag.get("reason")}
                        print(f"[BAR] {symbol} {info['bar_time']} action=NO_TRADE price={info['price']}")
                        log_action(cfg, info)

                # Between bars: keep it simple (polling)
                else:
                    # Optional: you can add intrabar checks here if desired.
                    pass

            time.sleep(cfg.poll_interval_sec)

        except Exception as e:
            print(f"[ERROR] main loop: {e}")
            time.sleep(cfg.poll_interval_sec)


if __name__ == "__main__":
    main()
