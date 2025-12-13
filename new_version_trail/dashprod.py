#!/usr/bin/env python3
"""
Trading Bot Admin Dashboard (FastAPI)

- Читає:
    actions_log.csv
    trades_log.csv
  які генерує твій бот.

- Показує:
    * Загальний PnL
    * PnL по кожному символу
    * Winrate
    * Відкриті позиції (по останній дії по символу)
    * Останні трейди
    * Останні дії бота

Запуск:
    uvicorn dashprod:app --reload --port 8000

Потім в браузері:
    http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List

ACTIONS_CSV = "actions_log.csv"
TRADES_CSV = "trades_log.csv"

app = FastAPI(title="Trading Bot Admin")

# CORS (про всяк випадок)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= Утиліти для CSV =========

def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Акуратне читання CSV:
    - якщо файлу немає → порожній DataFrame
    - якщо є криві рядки → пропускаємо (on_bad_lines="skip")
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        # Для старіших версій pandas без on_bad_lines
        df = pd.read_csv(path)

    return df


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Конвертація DataFrame → list[dict] з:
    - заміною NaN/NaT на None (щоб JSON не ламався)
    """
    if df.empty:
        return []

    # Замінюємо NaN/NaT на None
    df = df.where(pd.notna(df), None)

    # Переконуємось, що всі значення або Python-примітиви, або None
    records = df.to_dict(orient="records")
    return records


def load_actions() -> pd.DataFrame:
    """
    Читаємо actions_log.csv БЕЗ конвертації в Timestamp.
    bar_time лишається рядком (ISO), сортування по рядку для ISO працює ок.
    """
    df = safe_read_csv(ACTIONS_CSV)
    if df.empty:
        return df

    # Гарантуємо наявність базових колонок (якщо файл створений руками)
    base_cols = [
        "bar_time", "symbol", "action", "price", "position_was_open",
        "direction", "entry_price", "sl", "tp",
        "diag_reason", "exit_reason", "exit_price",
    ]
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    # Сортуємо по часу, якщо є
    if "bar_time" in df.columns:
        df = df.sort_values("bar_time")

    return df


def load_trades() -> pd.DataFrame:
    """
    Читаємо trades_log.csv.
    Теж без перетворення дат у Timestamp, працюємо як з рядками.
    """
    df = safe_read_csv(TRADES_CSV)
    if df.empty:
        return df

    base_cols = [
        "symbol", "entry_time", "exit_time", "direction",
        "entry_price", "exit_price", "pnl_abs", "pnl_pct",
        "reason", "size",
    ]
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    # Пробуємо привести числові колонки до float
    for num_col in ["entry_price", "exit_price", "pnl_abs", "pnl_pct", "size"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Сортуємо по exit_time, якщо є
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time")

    return df


# ========= API: SUMMARY =========

@app.get("/api/summary")
def get_summary():
    trades = load_trades()
    actions = load_actions()

    summary: Dict[str, Any] = {}

    # ---- Базова статистика по трейдам ----
    if trades.empty:
        summary["total_pnl_abs"] = 0.0
        summary["total_pnl_pct"] = 0.0
        summary["trades_count"] = 0
        summary["winrate"] = 0.0
        summary["pnl_by_symbol"] = []
    else:
        pnl_abs = pd.to_numeric(trades["pnl_abs"], errors="coerce")
        pnl_pct = pd.to_numeric(trades["pnl_pct"], errors="coerce")

        total_pnl_abs = float(pnl_abs.sum(skipna=True))
        # середній % на трейд (вже у %, не в долях)
        mean_pct = float(pnl_pct.mean(skipna=True) * 100.0) if not pnl_pct.dropna().empty else 0.0

        trades_count = int(len(trades))

        wins = trades[pnl_abs > 0]
        winrate = float(len(wins) / trades_count * 100.0) if trades_count > 0 else 0.0

        if "symbol" in trades.columns:
            pnl_by_symbol = (
                trades.groupby("symbol")["pnl_abs"]
                .sum()
                .reset_index()
                .rename(columns={"pnl_abs": "pnl_abs_sum"})
            )
            pnl_by_symbol_records = df_to_records(pnl_by_symbol)
        else:
            pnl_by_symbol_records = []

        summary["total_pnl_abs"] = total_pnl_abs
        summary["total_pnl_pct"] = mean_pct
        summary["trades_count"] = trades_count
        summary["winrate"] = winrate
        summary["pnl_by_symbol"] = pnl_by_symbol_records

    # ---- Відкриті позиції: по останній дії по кожному символу ----
    open_positions = []
    if not actions.empty and "symbol" in actions.columns:
        latest_by_symbol = actions.sort_values("bar_time").groupby("symbol").tail(1)

        for _, row in latest_by_symbol.iterrows():
            action = str(row.get("action", "") or "")
            # Якщо остання дія — CLOSE_*, вважаємо, що позиція закрита
            if action.startswith("CLOSE_"):
                continue

            open_positions.append(
                {
                    "symbol": row.get("symbol"),
                    "direction": row.get("direction"),
                    "entry_price": _to_float_safe(row.get("entry_price")),
                    "sl": _to_float_safe(row.get("sl")),
                    "tp": _to_float_safe(row.get("tp")),
                    "last_action": action,
                    "last_time": row.get("bar_time"),
                }
            )

    summary["open_positions"] = open_positions

    # ---- Останні дії ----
    if not actions.empty:
        last_actions = actions.sort_values("bar_time", ascending=False).head(10).copy()
        # Переконуємось, що немає NaN
        last_actions_records = df_to_records(last_actions)
        summary["last_actions"] = last_actions_records
    else:
        summary["last_actions"] = []

    # ---- Останні трейди ----
    if not trades.empty:
        last_trades = trades.sort_values("exit_time", ascending=False).head(10).copy()
        last_trades_records = df_to_records(last_trades)
        summary["last_trades"] = last_trades_records
    else:
        summary["last_trades"] = []

    return JSONResponse(summary)


def _to_float_safe(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ========= API: повні таблиці =========

@app.get("/api/trades")
def get_trades():
    trades = load_trades()
    if trades.empty:
        return []

    trades = trades.sort_values("exit_time", ascending=False)
    return df_to_records(trades)


@app.get("/api/actions")
def get_actions(limit: int = 100):
    actions = load_actions()
    if actions.empty:
        return []

    actions = actions.sort_values("bar_time", ascending=False).head(limit)
    return df_to_records(actions)


# ========= Головна HTML-сторінка =========

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8" />
    <title>Trading Bot Admin</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0b1120;
            color: #e5e7eb;
            margin: 0;
            padding: 0;
        }
        header {
            padding: 16px 24px;
            background: #020617;
            border-bottom: 1px solid #1f2937;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .title {
            font-size: 20px;
            font-weight: 600;
        }
        main {
            padding: 16px 24px 40px 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: #020617;
            border-radius: 16px;
            border: 1px solid #1f2937;
            padding: 16px;
        }
        .card h3 {
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #9ca3af;
        }
        .card .value {
            font-size: 22px;
            font-weight: 600;
        }
        .green { color: #22c55e; }
        .red { color: #ef4444; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            padding: 6px 8px;
            border-bottom: 1px solid #1f2937;
        }
        th {
            text-align: left;
            font-weight: 500;
            color: #9ca3af;
        }
        tr:hover td {
            background: #020617;
        }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin: 24px 0 8px 0;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            background: #111827;
            color: #9ca3af;
            border: 1px solid #1f2937;
        }
        .pill-long { border-color: #22c55e33; color: #22c55e; }
        .pill-short { border-color: #ef444433; color: #f97373; }
        .pill-close { border-color: #3b82f633; color: #60a5fa; }
        .muted {
            color: #6b7280;
        }
        .mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        .toolbar {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        button {
            background: #111827;
            border-radius: 999px;
            border: 1px solid #1f2937;
            color: #e5e7eb;
            font-size: 12px;
            padding: 6px 10px;
            cursor: pointer;
        }
        button:hover {
            background: #020617;
        }
        @media (max-width: 600px) {
            header, main {
                padding: 12px 16px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div>
            <div class="title">Trading Bot Admin</div>
            <div class="muted" style="font-size: 12px;">Futures mainnet data → TESTNET trades</div>
        </div>
        <div class="toolbar">
            <span id="last-refresh" class="muted" style="font-size: 12px;">Оновлення...</span>
            <button onclick="loadAll()">Оновити</button>
        </div>
    </header>
    <main>
        <section class="grid">
            <div class="card">
                <h3>Загальний PnL</h3>
                <div id="total-pnl" class="value mono">—</div>
                <div id="total-pnl-pct" class="muted" style="margin-top: 2px; font-size: 12px;">—</div>
            </div>
            <div class="card">
                <h3>Кількість трейдів</h3>
                <div id="trades-count" class="value">—</div>
                <div id="winrate" class="muted" style="margin-top: 2px; font-size: 12px;">Winrate: —</div>
            </div>
            <div class="card">
                <h3>Відкриті позиції</h3>
                <div id="open-positions-count" class="value">—</div>
                <div class="muted" style="margin-top: 2px; font-size: 12px;">По останніх діях</div>
            </div>
        </section>

        <section>
            <div class="section-title">PnL по символах</div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Символ</th>
                            <th class="mono">PnL</th>
                        </tr>
                    </thead>
                    <tbody id="pnl-by-symbol-body">
                        <tr><td colspan="2" class="muted">Немає даних</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <div class="section-title">Відкриті позиції</div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Символ</th>
                            <th>Напрям</th>
                            <th class="mono">Entry</th>
                            <th class="mono">SL</th>
                            <th class="mono">TP</th>
                            <th>Остання дія</th>
                            <th>Час</th>
                        </tr>
                    </thead>
                    <tbody id="open-positions-body">
                        <tr><td colspan="7" class="muted">Немає відкритих позицій</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <div class="section-title">Останні трейди</div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Символ</th>
                            <th>Напрям</th>
                            <th class="mono">Entry</th>
                            <th class="mono">Exit</th>
                            <th class="mono">PnL</th>
                            <th>Причина</th>
                            <th>Час виходу</th>
                        </tr>
                    </thead>
                    <tbody id="last-trades-body">
                        <tr><td colspan="7" class="muted">Немає трейдів</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <div class="section-title">Останні дії бота</div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Час</th>
                            <th>Символ</th>
                            <th>Дія</th>
                            <th class="mono">Ціна</th>
                            <th>Причина</th>
                        </tr>
                    </thead>
                    <tbody id="last-actions-body">
                        <tr><td colspan="5" class="muted">Немає даних</td></tr>
                    </tbody>
                </table>
            </div>
        </section>
    </main>

    <script>
        function fmtPnL(value) {
            if (value === null || value === undefined || isNaN(value)) return "—";
            const v = Number(value);
            const cls = v >= 0 ? "green" : "red";
            return `<span class="${cls}">` + v.toFixed(4) + `</span>`;
        }

        function fmtPct(value) {
            if (value === null || value === undefined || isNaN(value)) return "—";
            const v = Number(value);
            const cls = v >= 0 ? "green" : "red";
            return `<span class="${cls}">` + v.toFixed(2) + `%</span>`;
        }

        function fmtTime(value) {
            if (!value) return "—";
            const d = new Date(value);
            if (isNaN(d.getTime())) return value;
            return d.toLocaleString();
        }

        function pillForDirection(direction) {
            if (direction === "LONG") return '<span class="pill pill-long">LONG</span>';
            if (direction === "SHORT") return '<span class="pill pill-short">SHORT</span>';
            return '<span class="pill">—</span>';
        }

        function pillForAction(action) {
            if (!action) return '<span class="pill">—</span>';
            if (action.startsWith("OPEN_")) return '<span class="pill">OPEN</span>';
            if (action.startsWith("CLOSE_")) return '<span class="pill pill-close">' + action.replace("CLOSE_", "") + '</span>';
            if (action === "HOLD") return '<span class="pill">HOLD</span>';
            return '<span class="pill">' + action + '</span>';
        }

        async function loadAll() {
            try {
                const res = await fetch('/api/summary');
                const data = await res.json();

                // --- Загальні метрики ---
                document.getElementById("total-pnl").innerHTML = fmtPnL(data.total_pnl_abs);
                document.getElementById("total-pnl-pct").innerHTML =
                    "Середній PnL / трейд: " + fmtPct(data.total_pnl_pct);

                document.getElementById("trades-count").textContent = data.trades_count ?? 0;
                document.getElementById("winrate").textContent =
                    "Winrate: " + (data.winrate ? data.winrate.toFixed(1) + "%" : "—");

                document.getElementById("open-positions-count").textContent =
                    (data.open_positions || []).length;

                // --- PnL по символах ---
                const pnlBody = document.getElementById("pnl-by-symbol-body");
                pnlBody.innerHTML = "";
                if (!data.pnl_by_symbol || data.pnl_by_symbol.length === 0) {
                    pnlBody.innerHTML = '<tr><td colspan="2" class="muted">Немає даних</td></tr>';
                } else {
                    data.pnl_by_symbol.forEach(row => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${row.symbol}</td>
                            <td class="mono">${fmtPnL(row.pnl_abs_sum)}</td>
                        `;
                        pnlBody.appendChild(tr);
                    });
                }

                // --- Відкриті позиції ---
                const openBody = document.getElementById("open-positions-body");
                openBody.innerHTML = "";
                if (!data.open_positions || data.open_positions.length === 0) {
                    openBody.innerHTML = '<tr><td colspan="7" class="muted">Немає відкритих позицій</td></tr>';
                } else {
                    data.open_positions.forEach(pos => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${pos.symbol}</td>
                            <td>${pillForDirection(pos.direction)}</td>
                            <td class="mono">${pos.entry_price != null ? Number(pos.entry_price).toFixed(4) : "—"}</td>
                            <td class="mono">${pos.sl != null ? Number(pos.sl).toFixed(4) : "—"}</td>
                            <td class="mono">${pos.tp != null ? Number(pos.tp).toFixed(4) : "—"}</td>
                            <td>${pillForAction(pos.last_action)}</td>
                            <td class="muted">${fmtTime(pos.last_time)}</td>
                        `;
                        openBody.appendChild(tr);
                    });
                }

                // --- Останні трейди ---
                const lastTradesBody = document.getElementById("last-trades-body");
                lastTradesBody.innerHTML = "";
                if (!data.last_trades || data.last_trades.length === 0) {
                    lastTradesBody.innerHTML = '<tr><td colspan="7" class="muted">Немає трейдів</td></tr>';
                } else {
                    data.last_trades.forEach(trade => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${trade.symbol}</td>
                            <td>${pillForDirection(trade.direction)}</td>
                            <td class="mono">${trade.entry_price != null ? Number(trade.entry_price).toFixed(4) : "—"}</td>
                            <td class="mono">${trade.exit_price != null ? Number(trade.exit_price).toFixed(4) : "—"}</td>
                            <td class="mono">${fmtPnL(trade.pnl_abs)}</td>
                            <td class="muted">${trade.reason || "—"}</td>
                            <td class="muted">${fmtTime(trade.exit_time)}</td>
                        `;
                        lastTradesBody.appendChild(tr);
                    });
                }

                // --- Останні дії ---
                const lastActionsBody = document.getElementById("last-actions-body");
                lastActionsBody.innerHTML = "";
                if (!data.last_actions || data.last_actions.length === 0) {
                    lastActionsBody.innerHTML = '<tr><td colspan="5" class="muted">Немає даних</td></tr>';
                } else {
                    data.last_actions.forEach(act => {
                        const reason = act.diag_reason || act.exit_reason || "";
                        const price = act.price != null ? Number(act.price).toFixed(4) : "—";
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td class="muted">${fmtTime(act.bar_time)}</td>
                            <td>${act.symbol || ""}</td>
                            <td>${pillForAction(act.action)}</td>
                            <td class="mono">${price}</td>
                            <td class="muted">${reason}</td>
                        `;
                        lastActionsBody.appendChild(tr);
                    });
                }

                // Час оновлення
                const now = new Date();
                document.getElementById("last-refresh").textContent =
                    "Оновлено: " + now.toLocaleTimeString();

            } catch (e) {
                console.error(e);
                document.getElementById("last-refresh").textContent = "Помилка завантаження";
            }
        }

        loadAll();
        setInterval(loadAll, 15000);
    </script>
</body>
</html>
    """
    return HTMLResponse(html)
