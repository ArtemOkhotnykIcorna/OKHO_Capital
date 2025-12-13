#!/usr/bin/env python3
"""
Multi-Asset BTC & ETH Live Paper-Trading Bot (24/7 simulation, NO REAL TRADING)

- Тягне свічки BTCUSDT і ETHUSDT з публічного Binance API.
- Працює на закритті кожної нової свічки (interval, за замовчуванням 1m).
- Рахує фічі (momentum, EMA, Z-score, RV, ATR, ADX, volume MA).
- Вирішує: відкрити LONG / SHORT або нічого не робити.
- Для відкритої позиції:
    * на закритті бару — повна логіка (SL/TP/time-stop/трейлінг),
    * між барами — інтра-бар перевірка SL/TP/TP% по high/low (кожні poll_interval_sec).
- Логує всі дії та закриті трейди в ЄДИНІ CSV з колонкою symbol.

!!! НЕ ВИКОРИСТОВУЙ ЦЕ ДЛЯ РЕАЛЬНОЇ ТОРГІВЛІ !!!
"""

import time
import csv
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

import requests
import numpy as np
import pandas as pd

Signal = Literal["LONG", "SHORT", "WAIT"]

BINANCE_BASE_URL = "https://api.binance.com"

# Монети, з якими працює бот
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]


# =========================
# 1. Конфіг бота
# =========================

@dataclass
class BotConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    limit: int = 10000

    # Фічі
    ema_fast: int = 9
    ema_slow: int = 26
    mom_lookback: int = 16
    z_window: int = 40
    vol_window: int = 32

    atr_window: int = 14
    atr_mult_sl: float = 1.8     # трохи ширший стоп
    rr_ratio: float = 1.25        # ближчий тейк

    take_profit_pct: float = 0.005  # фіксований TP +0.5%

    min_realized_vol: float = 0.0015
    max_realized_vol: float = 0.0045  # новий верхній поріг волатильності

    # Z-score пороги
    z_long: float = 2.0      # LONG при z <= -z_long
    z_short: float = 2.5     # SHORT при z >= z_short
    min_momentum: float = 0.0015

    max_holding_bars: int = 16
    enable_trailing: bool = True
    trailing_atr_mult: float = 0.8

    position_size: float = 1.0
    poll_interval_sec: int = 15  # перевірка кожні 15 секунд

    # ADX / обʼєм / EMA
    adx_period: int = 10
    min_adx_trend: float = 18.0
    max_adx_trend: float = 32.0
    max_adx_for_mean_reversion: float = 25.0
    vol_ma_window: int = 15
    min_volume_ratio: float = 0.2
    max_dist_from_ema: float = 0.005
    max_ema_trend: float = 0.004  # ~0.4% розрив fast/slow EMA

    min_score: int = 3
    enable_confirmation_candle: bool = True

    # Єдині файли логів
    actions_log: str = "actions_log.csv"
    trades_log: str = "trades_log.csv"


# =========================
# 2. Структура позиції
# =========================

@dataclass
class Position:
    is_open: bool
    direction: Optional[Literal["LONG", "SHORT"]] = None
    entry_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    bars_in_trade: int = 0
    entry_time: Optional[pd.Timestamp] = None
    last_update_time: Optional[pd.Timestamp] = None
    entry_reason: Optional[str] = None
    size: float = 0.0


# =========================
# 3. Завантаження даних
# =========================

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Тягнемо свічки з публічного Binance REST API.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    float_cols = ["open", "high", "low", "close", "volume"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("close_time", inplace=True)

    return df


# =========================
# 4. Обчислення фіч
# =========================

def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Обчислення ADX без сторонніх бібліотек.
    Використовує EMA як наближення до Wilder smoothing.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    if "tr" not in df.columns:
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = df["tr"]

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def compute_features(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    df = df.copy()

    # Лог-доходність
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # Моментум
    df["momentum"] = df["close"] / df["close"].shift(cfg.mom_lookback) - 1.0

    # EMA fast/slow
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_trend"] = df["ema_fast"] - df["ema_slow"]

    # SMA + std для Z-score
    df["sma"] = df["close"].rolling(cfg.z_window).mean()
    df["sma_std"] = df["close"].rolling(cfg.z_window).std()
    df["z_score"] = (df["close"] - df["sma"]) / df["sma_std"]

    # Realized volatility
    df["rv"] = df["log_ret"].rolling(cfg.vol_window).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=True
    )

    # TR + ATR
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(cfg.atr_window).mean()

    # ADX
    df["adx"] = compute_adx(df, cfg.adx_period)

    # Volume MA
    df["vol_ma"] = df["volume"].rolling(cfg.vol_ma_window).mean()

    return df


# =========================
# 5. Логіка входу (entry) з фільтрами
# =========================

def decide_entry_signal(latest: pd.Series, cfg: BotConfig) -> Tuple[Signal, Dict]:
    price = float(latest["close"])
    momentum = float(latest["momentum"])
    ema_trend = float(latest["ema_trend"])
    z = float(latest["z_score"])
    rv = float(latest["rv"])
    atr = float(latest["atr"])

    volume = float(latest["volume"])
    vol_ma = float(latest.get("vol_ma", np.nan))
    adx = float(latest.get("adx", np.nan))
    ema_fast = float(latest["ema_fast"])
    candle_open = float(latest["open"])
    candle_close = float(latest["close"])

    info = {
        "price": price,
        "momentum": momentum,
        "ema_trend": ema_trend,
        "z_score": z,
        "realized_vol": rv,
        "volume": volume,
        "vol_ma": vol_ma,
        "adx": adx,
        "atr": atr,
    }

    # 0) Фільтр обʼєму
    if not np.isnan(vol_ma) and vol_ma > 0:
        if volume < cfg.min_volume_ratio * vol_ma:
            return "WAIT", {**info, "reason": "low_volume"}

    # 1) Волатильність: занадто низька або занадто висока → чекаємо
    if np.isnan(rv) or rv < cfg.min_realized_vol:
        return "WAIT", {**info, "reason": "low_volatility"}
    if cfg.max_realized_vol is not None and rv > cfg.max_realized_vol:
        return "WAIT", {**info, "reason": "too_much_volatility"}

    # 2) Mean-reversion по Z-score (LONG/SHORT окремі пороги)
    if not np.isnan(z):
        signal: Signal = "WAIT"
        reason = ""

        if z <= -cfg.z_long:
            signal = "LONG"
            reason = "extreme_oversold_z"
        elif z >= cfg.z_short:
            signal = "SHORT"
            reason = "extreme_overbought_z"

        if signal != "WAIT":
            # Фільтр відстані до EMA
            dist_from_ema = abs(price - ema_fast) / price
            info["dist_from_ema"] = dist_from_ema
            if dist_from_ema > cfg.max_dist_from_ema:
                return "WAIT", {**info, "reason": "overextended_from_ema_z"}

            # Mean-reversion тільки при не надто сильному тренді
            if not np.isnan(adx) and adx > cfg.max_adx_for_mean_reversion:
                return "WAIT", {**info, "reason": "no_mean_reversion_in_strong_trend"}

            # Confirmation candle
            if cfg.enable_confirmation_candle:
                if signal == "LONG" and not (candle_close > candle_open):
                    return "WAIT", {**info, "reason": "no_bullish_confirmation_z"}
                if signal == "SHORT" and not (candle_close < candle_open):
                    return "WAIT", {**info, "reason": "no_bearish_confirmation_z"}

            return signal, {**info, "reason": reason}

    # 3) Слабкий моментум → чекаємо
    if np.isnan(momentum) or abs(momentum) < cfg.min_momentum:
        return "WAIT", {**info, "reason": "weak_momentum"}

    # 4) Тренд-фолловінг + scoring
    candidate_signal: Signal = "WAIT"
    if ema_trend > 0 and momentum > 0:
        candidate_signal = "LONG"
    elif ema_trend < 0 and momentum < 0:
        candidate_signal = "SHORT"
    else:
        return "WAIT", {**info, "reason": "conflict_features"}

    # 4.1) ADX для тренду: в коридорі [min, max]
    if np.isnan(adx) or adx < cfg.min_adx_trend or adx > cfg.max_adx_trend:
        return "WAIT", {**info, "reason": "trend_adx_out_of_range"}

    # 4.2) Фільтр відстані до EMA
    dist_from_ema = abs(price - ema_fast) / price
    info["dist_from_ema"] = dist_from_ema
    if dist_from_ema > cfg.max_dist_from_ema:
        return "WAIT", {**info, "reason": "overextended_from_ema"}

    # 4.3) Не заходимо в занадто “розігнаний” тренд
    if abs(ema_trend) > cfg.max_ema_trend:
        return "WAIT", {**info, "reason": "ema_trend_too_strong"}

    # 4.4) Multi-factor score
    score = 0

    # фактор 1: напрямок моментуму
    if (candidate_signal == "LONG" and momentum > 0) or (
        candidate_signal == "SHORT" and momentum < 0
    ):
        score += 1

    # фактор 2: знак ema_trend
    if (candidate_signal == "LONG" and ema_trend > 0) or (
        candidate_signal == "SHORT" and ema_trend < 0
    ):
        score += 1

    # фактор 3: адекватна волатильність
    if not np.isnan(rv) and (cfg.min_realized_vol <= rv <= cfg.max_realized_vol):
        score += 1

    # фактор 4: сила тренду ADX
    if not np.isnan(adx) and (cfg.min_adx_trend <= adx <= cfg.max_adx_trend):
        score += 1

    info["score"] = score

    if score < cfg.min_score:
        return "WAIT", {**info, "reason": f"weak_score_{score}"}

    # 4.5) Confirmation candle
    if cfg.enable_confirmation_candle:
        if candidate_signal == "LONG" and not (candle_close > candle_open):
            return "WAIT", {**info, "reason": "no_bullish_confirmation"}
        if candidate_signal == "SHORT" and not (candle_close < candle_open):
            return "WAIT", {**info, "reason": "no_bearish_confirmation"}

    return candidate_signal, {**info, "reason": "trend_following_scored"}


# =========================
# 6. SL / TP
# =========================

def compute_sl_tp(direction: str, entry_price: float, atr: float,
                  cfg: BotConfig) -> Tuple[float, float, Dict]:
    if np.isnan(atr) or atr <= 0:
        atr = entry_price * 0.002  # fallback ~0.2%

    risk_points = cfg.atr_mult_sl * atr

    if direction == "LONG":
        sl = entry_price - risk_points
        tp = entry_price + risk_points * cfg.rr_ratio
    elif direction == "SHORT":
        sl = entry_price + risk_points
        tp = entry_price - risk_points * cfg.rr_ratio
    else:
        raise ValueError("direction must be 'LONG' or 'SHORT'")

    meta = {"atr": atr, "risk_points": risk_points, "rr_ratio": cfg.rr_ratio}
    return sl, tp, meta


# =========================
# 7. Оновлення позиції (exit + trailing) НА ЗАКРИТТІ БАРУ
# =========================

def update_position_with_bar(
    position: Position,
    latest: pd.Series,
    cfg: BotConfig,
) -> Tuple[Position, Optional[Dict]]:
    if not position.is_open:
        return position, None

    high = float(latest["high"])
    low = float(latest["low"])
    close_price = float(latest["close"])
    atr = float(latest["atr"])
    now = latest.name  # close_time Timestamp

    exit_event = None
    exit_price = close_price  # дефолт

    # 1. SL / TP по рівнях (по high/low)
    if position.direction == "LONG":
        if low <= position.sl:
            exit_event = {
                "type": "CLOSE_SL",
                "reason": "hit_stop_loss",
            }
            exit_price = position.sl
        elif high >= position.tp:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": "hit_take_profit",
            }
            exit_price = position.tp

    elif position.direction == "SHORT":
        if high >= position.sl:
            exit_event = {
                "type": "CLOSE_SL",
                "reason": "hit_stop_loss",
            }
            exit_price = position.sl
        elif low <= position.tp:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": "hit_take_profit",
            }
            exit_price = position.tp

    # 1.5. Фіксований тейк-профіт по % (по high/low)
    if exit_event is None and cfg.take_profit_pct is not None:
        if position.direction == "LONG":
            max_price = high
            pnl_pct = (max_price - position.entry_price) / position.entry_price
            target_price = position.entry_price * (1 + cfg.take_profit_pct)
        elif position.direction == "SHORT":
            min_price = low
            pnl_pct = (position.entry_price - min_price) / position.entry_price
            target_price = position.entry_price * (1 - cfg.take_profit_pct)
        else:
            pnl_pct = 0.0
            target_price = close_price

        if pnl_pct >= cfg.take_profit_pct:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": f"hit_fixed_tp_{cfg.take_profit_pct*100:.2f}pct",
            }
            exit_price = target_price

    # 2. Time-stop (тільки на закритті бару)
    if exit_event is None:
        position.bars_in_trade += 1
        if position.bars_in_trade >= cfg.max_holding_bars:
            exit_event = {
                "type": "CLOSE_TIME",
                "reason": "max_holding_bars_reached",
            }
            exit_price = close_price

    # 3. Трейлінг-стоп
    if exit_event is None and cfg.enable_trailing and not np.isnan(atr) and atr > 0:
        if position.direction == "LONG":
            unrealized = close_price - position.entry_price
            if unrealized > cfg.trailing_atr_mult * atr:
                new_sl = close_price - cfg.trailing_atr_mult * atr
                position.sl = max(position.sl, new_sl)
        elif position.direction == "SHORT":
            unrealized = position.entry_price - close_price
            if unrealized > cfg.trailing_atr_mult * atr:
                new_sl = close_price + cfg.trailing_atr_mult * atr
                position.sl = min(position.sl, new_sl)

    # 4. Закриття
    if exit_event is not None:
        position.is_open = False
        position.last_update_time = now
        exit_event["bar_time"] = now.isoformat()
        exit_event["price"] = exit_price
        return position, exit_event

    position.last_update_time = now
    return position, None


# =========================
# 7.5. Інтра-бар перевірка виходу (кожні poll_interval_sec)
# =========================

def check_intrabar_exit(
    position: Position,
    latest: pd.Series,
    cfg: BotConfig,
) -> Tuple[Position, Optional[Dict]]:
    """
    Інтра-бар перевірка: дивимось по high/low (SL/TP/fixed TP/трейлінг),
    але НЕ чіпаємо bars_in_trade і НЕ закриваємо по time-stop.

    Викликається всередині бару (кожні poll_interval_sec секунд),
    коли нового бару ще немає, але позиція відкрита.
    """
    if not position.is_open:
        return position, None

    high = float(latest["high"])
    low = float(latest["low"])
    close_price = float(latest["close"])
    atr = float(latest["atr"])
    now = latest.name  # час останнього бару (для логів)

    exit_event = None
    exit_price = close_price

    # 1. SL / TP по рівнях
    if position.direction == "LONG":
        if low <= position.sl:
            exit_event = {
                "type": "CLOSE_SL",
                "reason": "hit_stop_loss_intrabar",
            }
            exit_price = position.sl
        elif high >= position.tp:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": "hit_take_profit_intrabar",
            }
            exit_price = position.tp

    elif position.direction == "SHORT":
        if high >= position.sl:
            exit_event = {
                "type": "CLOSE_SL",
                "reason": "hit_stop_loss_intrabar",
            }
            exit_price = position.sl
        elif low <= position.tp:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": "hit_take_profit_intrabar",
            }
            exit_price = position.tp

    # 1.5. Фіксований тейк-профіт по %
    if exit_event is None and cfg.take_profit_pct is not None:
        if position.direction == "LONG":
            max_price = high
            pnl_pct = (max_price - position.entry_price) / position.entry_price
            target_price = position.entry_price * (1 + cfg.take_profit_pct)
        elif position.direction == "SHORT":
            min_price = low
            pnl_pct = (position.entry_price - min_price) / position.entry_price
            target_price = position.entry_price * (1 - cfg.take_profit_pct)
        else:
            pnl_pct = 0.0
            target_price = close_price

        if pnl_pct >= cfg.take_profit_pct:
            exit_event = {
                "type": "CLOSE_TP",
                "reason": f"hit_fixed_tp_intrabar_{cfg.take_profit_pct*100:.2f}pct",
            }
            exit_price = target_price

    # 2. Трейлінг-стоп (оновлюємо і всередині бару)
    if exit_event is None and cfg.enable_trailing and not np.isnan(atr) and atr > 0:
        if position.direction == "LONG":
            unrealized = close_price - position.entry_price
            if unrealized > cfg.trailing_atr_mult * atr:
                new_sl = close_price - cfg.trailing_atr_mult * atr
                position.sl = max(position.sl, new_sl)
        elif position.direction == "SHORT":
            unrealized = position.entry_price - close_price
            if unrealized > cfg.trailing_atr_mult * atr:
                new_sl = close_price + cfg.trailing_atr_mult * atr
                position.sl = min(position.sl, new_sl)

    # 3. Якщо спрацював вихід — закриваємо позицію
    if exit_event is not None:
        position.is_open = False
        position.last_update_time = now
        exit_event["bar_time"] = now.isoformat()
        exit_event["price"] = exit_price
        return position, exit_event

    position.last_update_time = now
    return position, None


# =========================
# 8. Логіка "один бар"
# =========================

def process_new_bar(
    df: pd.DataFrame,
    position: Position,
    cfg: BotConfig,
) -> Tuple[Position, Dict]:
    """
    Викликається коли з'явилась НОВА свічка.
    df — вже з фічами, беремо останній рядок як "закритий бар".
    """
    latest = df.iloc[-1]
    bar_time = latest.name
    price = float(latest["close"])

    info: Dict = {
        "bar_time": bar_time.isoformat(),
        "price": price,
    }

    # Якщо позиція є → оновлюємо
    if position.is_open:
        position, exit_event = update_position_with_bar(position, latest, cfg)
        info["position_before"] = "OPEN"
        info["direction"] = position.direction
        info["entry_price"] = position.entry_price
        info["sl"] = position.sl
        info["tp"] = position.tp
        info["bars_in_trade"] = position.bars_in_trade

        if exit_event is not None:
            info["action"] = exit_event["type"]
            info["exit_reason"] = exit_event["reason"]
            info["exit_price"] = exit_event["price"]
            info["exit_bar_time"] = exit_event.get("bar_time")
        else:
            info["action"] = "HOLD"

        return position, info

    # Якщо позиції немає → шукаємо вхід
    signal, diag = decide_entry_signal(latest, cfg)
    info.update({f"diag_{k}": v for k, v in diag.items()})

    if signal in ("LONG", "SHORT"):
        atr = float(latest["atr"])
        sl, tp, meta = compute_sl_tp(signal, price, atr, cfg)

        position = Position(
            is_open=True,
            direction=signal,
            entry_price=price,
            sl=sl,
            tp=tp,
            bars_in_trade=0,
            entry_time=bar_time,
            last_update_time=bar_time,
            entry_reason=diag.get("reason"),
            size=cfg.position_size,
        )

        info["action"] = f"OPEN_{signal}"
        info["entry_reason"] = diag.get("reason")
        info["entry_price"] = price
        info["sl"] = sl
        info["tp"] = tp
        info.update({f"meta_{k}": v for k, v in meta.items()})
    else:
        info["action"] = "NO_TRADE"

    return position, info


# =========================
# 9. Логування
# =========================

def ensure_csv_headers(filepath: str, fieldnames):
    exists = os.path.exists(filepath)
    if not exists:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def log_action(cfg: BotConfig, info: Dict):
    fieldnames = sorted(info.keys())
    ensure_csv_headers(cfg.actions_log, fieldnames)
    with open(cfg.actions_log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(info)


def log_trade(cfg: BotConfig, position: Position, exit_info: Dict):
    """
    Лог закритого трейду у trades_log.csv
    Викликається, коли action = CLOSE_SL / CLOSE_TP / CLOSE_TIME.
    """
    symbol = exit_info.get("symbol", "UNKNOWN")

    entry_price = position.entry_price
    exit_price = exit_info.get("exit_price")
    direction = position.direction
    size = position.size

    if direction == "LONG":
        pnl_abs = (exit_price - entry_price) * size
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_abs = (entry_price - exit_price) * size
        pnl_pct = (entry_price - exit_price) / entry_price

    row = {
        "symbol": symbol,
        "entry_time": position.entry_time.isoformat() if position.entry_time else "",
        "exit_time": exit_info.get("exit_bar_time", ""),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "reason": exit_info.get("exit_reason", ""),
        "size": size,
    }

    # ФІКСОВАНИЙ порядок колонок
    fieldnames = [
        "symbol",
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "pnl_abs",
        "pnl_pct",
        "reason",
        "size",
    ]

    ensure_csv_headers(cfg.trades_log, fieldnames)
    with open(cfg.trades_log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# =========================
# 10. main-loop 24/7 (BTC + ETH)
# =========================

def main():
    cfg = BotConfig()

    print("=== Multi-Asset Live Paper-Trading Bot ===")
    print(f"Symbols: {SYMBOLS}, interval: {cfg.interval}")
    print("poll_interval_sec:", cfg.poll_interval_sec)
    print("!!! Це симуляція. Жодних реальних ордерів не відправляється.\n")

    # окремий стейт для кожного символу
    positions: Dict[str, Position] = {symbol: Position(is_open=False) for symbol in SYMBOLS}
    last_bar_times: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in SYMBOLS}

    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch_klines(symbol, cfg.interval, cfg.limit)
                df = compute_features(df, cfg)

                latest_time = df.index[-1]
                latest_row = df.iloc[-1]

                # 1) Якщо зʼявився НОВИЙ бар → стандартна логіка (входи + виходи по бару)
                if last_bar_times[symbol] is None or latest_time > last_bar_times[symbol]:
                    last_bar_times[symbol] = latest_time

                    position_before = positions[symbol].is_open
                    positions[symbol], info = process_new_bar(df, positions[symbol], cfg)
                    info["symbol"] = symbol
                    info["position_was_open"] = position_before

                    # Лог у консоль
                    print(f"[BAR] [{symbol}] [{info['bar_time']}] action={info['action']}, price={info['price']}")

                    # Лог у файл з усіма діями
                    log_action(cfg, info)

                    # Якщо трейд закрився — лог трейду
                    if info["action"] in ("CLOSE_SL", "CLOSE_TP", "CLOSE_TIME"):
                        info["exit_bar_time"] = info.get("exit_bar_time", info["bar_time"])
                        log_trade(cfg, positions[symbol], info)

                # 2) Нового бару немає, але позиція відкрита → інтра-бар перевірка SL/TP кожні poll_interval_sec
                else:
                    if positions[symbol].is_open:
                        position_before = positions[symbol].is_open
                        positions[symbol], exit_event = check_intrabar_exit(
                            positions[symbol],
                            latest_row,
                            cfg,
                        )

                        if exit_event is not None:
                            # Формуємо info для логування (мінімально необхідне)
                            info = {
                                "bar_time": exit_event["bar_time"],
                                "price": exit_event["price"],
                                "symbol": symbol,
                                "position_was_open": position_before,
                                "action": exit_event["type"],
                                "exit_reason": exit_event["reason"],
                                "exit_price": exit_event["price"],
                                "direction": positions[symbol].direction,
                                "entry_price": positions[symbol].entry_price,
                                "sl": positions[symbol].sl,
                                "tp": positions[symbol].tp,
                            }

                            print(f"[INTRABAR] [{symbol}] [{info['bar_time']}] action={info['action']}, price={info['price']}")
                            log_action(cfg, info)

                            # Для log_trade потрібен exit_bar_time
                            info["exit_bar_time"] = exit_event["bar_time"]
                            log_trade(cfg, positions[symbol], info)

            # Чекаємо до наступної ітерації
            time.sleep(cfg.poll_interval_sec)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(cfg.poll_interval_sec)


if __name__ == "__main__":
    main()
