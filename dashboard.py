# dashboard.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
from pandas.errors import EmptyDataError
from pathlib import Path

TRADES_FILE = Path("trades_log.csv")
ACTIONS_FILE = Path("actions_log.csv")

app = FastAPI(title="Paper Bot Dashboard")


# =========================
# Лоадери CSV
# =========================

def load_trades() -> pd.DataFrame:
    """
    Завантаження trades_log.csv.
    Підтримує:
    - варіант без header (рядок одразу: direction,entry_price,...,symbol)
    - варіант з дивними назвами колонок, але 10 штук.
    """
    if not TRADES_FILE.exists():
        raise FileNotFoundError("trades_log.csv not found")

    try:
        df = pd.read_csv(TRADES_FILE)
    except EmptyDataError:
        return pd.DataFrame()

    expected_cols = [
        "direction",
        "entry_price",
        "entry_time",
        "exit_price",
        "exit_time",
        "pnl_abs",
        "pnl_pct",
        "reason",
        "size",
        "symbol",
    ]

    # Якщо немає pnl_pct, але рівно 10 колонок → трактуємо як файл без заголовка
    if "pnl_pct" not in df.columns and df.shape[1] == 10:
        df = pd.read_csv(TRADES_FILE, header=None)
        df.columns = expected_cols
    elif df.shape[1] == 10 and set(expected_cols).issubset(set(df.columns)) is False:
        # 10 колонок, але дивні назви — теж форсимо наш порядок
        df.columns = expected_cols

    if df.empty:
        return df

    # Час
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

    # PnL
    if "pnl_pct" in df.columns:
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    if "pnl_abs" in df.columns:
        df["pnl_abs"] = pd.to_numeric(df["pnl_abs"], errors="coerce")

    # Ціни
    if "entry_price" in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    if "exit_price" in df.columns:
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")

    return df


def load_actions() -> pd.DataFrame:
    """
    Завантаження actions_log.csv з пропуском битих рядків.
    """
    if not ACTIONS_FILE.exists():
        raise FileNotFoundError("actions_log.csv not found")
    try:
        df = pd.read_csv(
            ACTIONS_FILE,
            engine="python",
            on_bad_lines="skip",
        )
    except EmptyDataError:
        return pd.DataFrame()

    if df.empty:
        return df

    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")

    # базові числові поля
    for col in ["price", "size", "bars_in_trade", "sl", "tp", "entry_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str)

    return df


# =========================
# Summary по трейдах
# =========================

def compute_summary(df: pd.DataFrame) -> dict:
    if df.empty or "pnl_pct" not in df.columns:
        return {
            "n_trades": 0,
            "winrate": 0,
            "total_pnl_pct": 0,
            "avg_pnl_pct": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
        }

    df_valid = df.dropna(subset=["pnl_pct"])
    if df_valid.empty:
        return {
            "n_trades": 0,
            "winrate": 0,
            "total_pnl_pct": 0,
            "avg_pnl_pct": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
        }

    n = len(df_valid)
    wins = df_valid[df_valid["pnl_pct"] > 0]
    losses = df_valid[df_valid["pnl_pct"] <= 0]

    total_pnl_pct = df_valid["pnl_pct"].sum()
    winrate = len(wins) / n if n > 0 else 0
    avg_pnl_pct = df_valid["pnl_pct"].mean()
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0

    return {
        "n_trades": n,
        "winrate": winrate,
        "total_pnl_pct": total_pnl_pct,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }


# =========================
# Активні позиції: програємо стейт з actions_log
# =========================

def compute_open_positions(df_actions: pd.DataFrame) -> list:
    """
    Розбираємо actions_log.csv по часу і програємо стан по кожному symbol.

    Логіка:
      - action починається з "OPEN_" → відкрили позицію
      - "HOLD" → позиція відкрита, оновлюємо last_price, bars_in_trade, sl/tp
      - action починається з "CLOSE_" → позиція закрилась
      - інші дії ("NO_TRADE") просто оновлюють last_price, якщо позиція вже була відкрита

    На виході: список відкритих позицій зі всіма параметрами.
    """
    if df_actions.empty or "symbol" not in df_actions.columns:
        return []

    # сортуємо по часу
    if "bar_time" in df_actions.columns:
        df_actions = df_actions.sort_values("bar_time")

    # стан по кожному символу
    states = {}  # symbol -> dict

    def to_float(val):
        if val is None or pd.isna(val):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    for _, row in df_actions.iterrows():
        symbol = row.get("symbol")
        if symbol is None or pd.isna(symbol):
            continue
        symbol = str(symbol)

        action_raw = row.get("action", "")
        action = str(action_raw) if action_raw is not None else ""

        bar_time = row.get("bar_time", None)

        price = to_float(row.get("price"))
        entry_price_col = to_float(row.get("entry_price"))
        sl_col = to_float(row.get("sl"))
        tp_col = to_float(row.get("tp"))
        size_col = to_float(row.get("size"))
        bars_col = row.get("bars_in_trade", None)
        try:
            bars_in_trade_col = int(bars_col) if bars_col is not None and not pd.isna(bars_col) else None
        except (TypeError, ValueError):
            bars_in_trade_col = None

        direction_row = row.get("direction", None)
        direction = str(direction_row) if direction_row not in (None, float("nan")) else ""

        if symbol not in states:
            states[symbol] = {
                "is_open": False,
                "direction": "",
                "entry_price": None,
                "last_price": None,
                "size": 1.0,
                "bars_in_trade": None,
                "sl": None,
                "tp": None,
                "entry_time": None,
                "last_bar_time": None,
            }

        st = states[symbol]

        # OPEN
        if action.startswith("OPEN_"):
            st["is_open"] = True
            # напрям: з direction або з назви action
            if "LONG" in action:
                st["direction"] = "LONG"
            elif "SHORT" in action:
                st["direction"] = "SHORT"
            elif direction.upper() in ("LONG", "SHORT"):
                st["direction"] = direction.upper()
            else:
                st["direction"] = ""

            st["entry_price"] = entry_price_col if entry_price_col is not None else price
            st["last_price"] = price
            st["size"] = size_col if size_col is not None else 1.0
            st["bars_in_trade"] = bars_in_trade_col
            st["sl"] = sl_col
            st["tp"] = tp_col
            st["entry_time"] = bar_time
            st["last_bar_time"] = bar_time

        # HOLD: оновлюємо тільки якщо позиція вже відкрита
        elif action == "HOLD":
            if st["is_open"]:
                if price is not None:
                    st["last_price"] = price
                if bars_in_trade_col is not None:
                    st["bars_in_trade"] = bars_in_trade_col
                if sl_col is not None:
                    st["sl"] = sl_col
                if tp_col is not None:
                    st["tp"] = tp_col
                st["last_bar_time"] = bar_time

        # CLOSE: позиція закривається
        elif action.startswith("CLOSE_"):
            st["is_open"] = False
            st["last_bar_time"] = bar_time
            # entry_price не чіпаємо, але позиція вважається закритою

        else:
            # Наприклад, NO_TRADE – просто оновлюємо останню ціну, якщо позиція відкрита
            if st["is_open"] and price is not None:
                st["last_price"] = price
                st["last_bar_time"] = bar_time

    # Формуємо список відкритих позицій
    open_positions = []

    for symbol, st in states.items():
        if not st["is_open"]:
            continue

        entry_price = st["entry_price"]
        last_price = st["last_price"]
        direction = st["direction"]
        size = st["size"] if st["size"] is not None else 1.0

        pnl_abs = None
        pnl_pct = None
        if entry_price is not None and last_price is not None and direction in ("LONG", "SHORT"):
            if direction == "LONG":
                pnl_abs = (last_price - entry_price) * size
                pnl_pct = (last_price - entry_price) / entry_price
            else:
                pnl_abs = (entry_price - last_price) * size
                pnl_pct = (entry_price - last_price) / entry_price

        entry_time = st["entry_time"]
        last_bar_time = st["last_bar_time"]

        if isinstance(entry_time, pd.Timestamp) and pd.notna(entry_time):
            entry_time_str = entry_time.isoformat()
        else:
            entry_time_str = str(entry_time) if entry_time is not None and not pd.isna(entry_time) else None

        if isinstance(last_bar_time, pd.Timestamp) and pd.notna(last_bar_time):
            last_bar_time_str = last_bar_time.isoformat()
        else:
            last_bar_time_str = str(last_bar_time) if last_bar_time is not None and not pd.isna(last_bar_time) else None

        open_positions.append({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "last_price": last_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "size": size,
            "bars_in_trade": st["bars_in_trade"],
            "sl": st["sl"],
            "tp": st["tp"],
            "entry_time": entry_time_str,
            "last_bar_time": last_bar_time_str,
        })

    return open_positions


# =========================
# HTML-дашборд
# =========================

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8" />
        <title>Paper Bot Dashboard</title>
        <style>
            body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                   max-width: 1100px; margin: 20px auto; padding: 0 12px; background: #0b1120; color: #e5e7eb; }
            h1 { font-size: 24px; margin-bottom: 8px; }
            h2 { margin-top: 24px; }
            .card { background: #020617; border-radius: 12px; padding: 16px; margin-bottom: 16px;
                    border: 1px solid #1f2937; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
            .stat { font-size: 14px; }
            .label { color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
            .value { font-size: 18px; font-weight: 600; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th, td { padding: 6px 8px; border-bottom: 1px solid #1f2937; }
            th { text-align: left; color: #9ca3af; font-size: 12px; }
            tr:nth-child(even) { background: #020617; }
            .tag { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 11px; }
            .tag-win { background: #064e3b; color: #bbf7d0; }
            .tag-loss { background: #450a0a; color: #fecaca; }
            .dir-long { color: #22c55e; font-weight: 600; }
            .dir-short { color: #f97316; font-weight: 600; }
            .small { font-size: 11px; color: #9ca3af; }
            button { background: #1f2937; color: #e5e7eb; border-radius: 999px; border: 1px solid #374151;
                     padding: 6px 12px; font-size: 12px; cursor: pointer; }
            button:hover { background: #111827; }
            .tabs { display:flex; gap:8px; margin-bottom:16px; }
            .tab-btn { padding: 6px 14px; border-radius: 999px; border:1px solid #374151;
                       background:#020617; color:#e5e7eb; font-size:13px; cursor:pointer; }
            .tab-btn.active { background:#e5e7eb; color:#020617; border-color:#e5e7eb; }
            code { background:#020617; padding:2px 4px; border-radius:4px; }
        </style>
    </head>
    <body>
        <h1>Paper Bot Dashboard</h1>
        <p class="small">
            Статистика та логи з <code>trades_log.csv</code> і <code>actions_log.csv</code>.
        </p>

        <div class="tabs">
            <button id="tab-btn-trades" class="tab-btn active" onclick="setTab('trades')">Торги</button>
            <button id="tab-btn-actions" class="tab-btn" onclick="setTab('actions')">Логи дій</button>
        </div>

        <!-- Торги -->
        <div id="tab-trades">
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2>Загальна статистика (усі символи)</h2>
                    <button onclick="refreshAll()">Оновити</button>
                </div>
                <div id="summary" class="grid">
                    <div class="stat">
                        <div class="label">Кількість угод</div>
                        <div class="value" id="s-n-trades">-</div>
                    </div>
                    <div class="stat">
                        <div class="label">Winrate</div>
                        <div class="value" id="s-winrate">-</div>
                    </div>
                    <div class="stat">
                        <div class="label">Сумарний PnL</div>
                        <div class="value" id="s-total-pnl">-</div>
                        <div class="small">сума PnL% по всіх угодах</div>
                    </div>
                    <div class="stat">
                        <div class="label">Середній PnL / угода</div>
                        <div class="value" id="s-avg-pnl">-</div>
                    </div>
                    <div class="stat">
                        <div class="label">Середній WIN</div>
                        <div class="value" id="s-avg-win">-</div>
                    </div>
                    <div class="stat">
                        <div class="label">Середній LOSS</div>
                        <div class="value" id="s-avg-loss">-</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Активні позиції</h2>
                <div class="small" id="open-pos-status"></div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Токен</th>
                                <th>Напрям</th>
                                <th>Вхід</th>
                                <th>Остання ціна</th>
                                <th>PnL %</th>
                                <th>PnL $</th>
                                <th>Size</th>
                                <th>Bars</th>
                                <th>SL</th>
                                <th>TP</th>
                                <th>Час входу</th>
                                <th>Останній бар</th>
                            </tr>
                        </thead>
                        <tbody id="open-pos-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>Останні угоди</h2>
                <div class="small" id="last-update-trades"></div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Вихід</th>
                                <th>Токен</th>
                                <th>Напрям</th>
                                <th>Вхід → вихід</th>
                                <th>PnL %</th>
                                <th>Причина</th>
                            </tr>
                        </thead>
                        <tbody id="trades-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Логи -->
        <div id="tab-actions" style="display:none;">
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2>Логи дій (actions_log.csv)</h2>
                    <button onclick="loadActions()">Оновити логи</button>
                </div>
                <div class="small" id="last-update-actions"></div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Час бару</th>
                                <th>Токен</th>
                                <th>Дія</th>
                                <th>Ціна</th>
                                <th>Статус позиції</th>
                                <th>Причина</th>
                            </tr>
                        </thead>
                        <tbody id="actions-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function setTab(tab) {
                const tradesDiv = document.getElementById("tab-trades");
                const actionsDiv = document.getElementById("tab-actions");
                const btnTrades = document.getElementById("tab-btn-trades");
                const btnActions = document.getElementById("tab-btn-actions");

                if (tab === "trades") {
                    tradesDiv.style.display = "";
                    actionsDiv.style.display = "none";
                    btnTrades.classList.add("active");
                    btnActions.classList.remove("active");
                } else {
                    tradesDiv.style.display = "none";
                    actionsDiv.style.display = "";
                    btnTrades.classList.remove("active");
                    btnActions.classList.add("active");
                }
            }

            async function fetchJSON(url) {
                const resp = await fetch(url);
                if (!resp.ok) {
                    throw new Error("HTTP " + resp.status);
                }
                return await resp.json();
            }

            // ===== Торги =====

            async function loadSummary() {
                try {
                    const data = await fetchJSON("/api/summary");
                    document.getElementById("s-n-trades").innerText = data.n_trades;
                    document.getElementById("s-winrate").innerText = (data.winrate * 100).toFixed(1) + "%";
                    document.getElementById("s-total-pnl").innerText = (data.total_pnl_pct * 100).toFixed(2) + "%";
                    document.getElementById("s-avg-pnl").innerText = (data.avg_pnl_pct * 100).toFixed(3) + "%";
                    document.getElementById("s-avg-win").innerText = (data.avg_win_pct * 100).toFixed(3) + "%";
                    document.getElementById("s-avg-loss").innerText = (data.avg_loss_pct * 100).toFixed(3) + "%";
                } catch (e) {
                    console.error(e);
                    document.getElementById("s-n-trades").innerText = "error";
                }
            }

            async function loadOpenPositions() {
                try {
                    const data = await fetchJSON("/api/open_positions");
                    const tbody = document.getElementById("open-pos-body");
                    const status = document.getElementById("open-pos-status");
                    tbody.innerHTML = "";

                    if (!data.positions || data.positions.length === 0) {
                        status.textContent = "Немає активних позицій";
                        return;
                    }

                    status.textContent = "Активних позицій: " + data.positions.length;

                    data.positions.forEach(p => {
                        const tr = document.createElement("tr");

                        const tdSymbol = document.createElement("td");
                        tdSymbol.textContent = p.symbol || "-";
                        tr.appendChild(tdSymbol);

                        const tdDir = document.createElement("td");
                        if (p.direction === "LONG") {
                            tdDir.innerHTML = '<span class="dir-long">LONG</span>';
                        } else if (p.direction === "SHORT") {
                            tdDir.innerHTML = '<span class="dir-short">SHORT</span>';
                        } else {
                            tdDir.textContent = p.direction || "";
                        }
                        tr.appendChild(tdDir);

                        const tdEntry = document.createElement("td");
                        tdEntry.textContent = p.entry_price !== null && p.entry_price !== undefined
                            ? p.entry_price.toFixed(2)
                            : "-";
                        tr.appendChild(tdEntry);

                        const tdLast = document.createElement("td");
                        tdLast.textContent = p.last_price !== null && p.last_price !== undefined
                            ? p.last_price.toFixed(2)
                            : "-";
                        tr.appendChild(tdLast);

                        const tdPnlPct = document.createElement("td");
                        const tagPct = document.createElement("span");
                        if (p.pnl_pct !== null && p.pnl_pct !== undefined) {
                            tagPct.className = p.pnl_pct >= 0 ? "tag tag-win" : "tag tag-loss";
                            tagPct.textContent = (p.pnl_pct * 100).toFixed(2) + "%";
                        } else {
                            tagPct.textContent = "-";
                        }
                        tdPnlPct.appendChild(tagPct);
                        tr.appendChild(tdPnlPct);

                        const tdPnlAbs = document.createElement("td");
                        const tagAbs = document.createElement("span");
                        if (p.pnl_abs !== null && p.pnl_abs !== undefined) {
                            tagAbs.className = p.pnl_abs >= 0 ? "tag tag-win" : "tag tag-loss";
                            tagAbs.textContent = p.pnl_abs.toFixed(4);
                        } else {
                            tagAbs.textContent = "-";
                        }
                        tdPnlAbs.appendChild(tagAbs);
                        tr.appendChild(tdPnlAbs);

                        const tdSize = document.createElement("td");
                        tdSize.textContent = p.size !== null && p.size !== undefined
                            ? p.size.toFixed(4)
                            : "-";
                        tr.appendChild(tdSize);

                        const tdBars = document.createElement("td");
                        tdBars.textContent = p.bars_in_trade !== null && p.bars_in_trade !== undefined
                            ? p.bars_in_trade
                            : "-";
                        tr.appendChild(tdBars);

                        const tdSl = document.createElement("td");
                        tdSl.textContent = p.sl !== null && p.sl !== undefined
                            ? p.sl.toFixed(2)
                            : "-";
                        tr.appendChild(tdSl);

                        const tdTp = document.createElement("td");
                        tdTp.textContent = p.tp !== null && p.tp !== undefined
                            ? p.tp.toFixed(2)
                            : "-";
                        tr.appendChild(tdTp);

                        const tdEntryTime = document.createElement("td");
                        tdEntryTime.textContent = p.entry_time || "-";
                        tdEntryTime.className = "small";
                        tr.appendChild(tdEntryTime);

                        const tdLastTime = document.createElement("td");
                        tdLastTime.textContent = p.last_bar_time || "-";
                        tdLastTime.className = "small";
                        tr.appendChild(tdLastTime);

                        tbody.appendChild(tr);
                    });

                } catch (e) {
                    console.error(e);
                    document.getElementById("open-pos-status").textContent = "Помилка завантаження активних позицій";
                }
            }

            async function loadTrades() {
                try {
                    const data = await fetchJSON("/api/trades?limit=30");
                    const tbody = document.getElementById("trades-body");
                    tbody.innerHTML = "";
                    data.trades.forEach(t => {
                        const tr = document.createElement("tr");

                        const tdTime = document.createElement("td");
                        tdTime.textContent = t.exit_time || "-";
                        tdTime.className = "small";
                        tr.appendChild(tdTime);

                        const tdSymbol = document.createElement("td");
                        tdSymbol.textContent = t.symbol || "-";
                        tr.appendChild(tdSymbol);

                        const tdDir = document.createElement("td");
                        tdDir.innerHTML = t.direction === "LONG"
                            ? '<span class="dir-long">LONG</span>'
                            : '<span class="dir-short">SHORT</span>';
                        tr.appendChild(tdDir);

                        const tdPrices = document.createElement("td");
                        const ep = (t.entry_price ?? 0).toFixed(2);
                        const xp = (t.exit_price ?? 0).toFixed(2);
                        tdPrices.textContent = ep + " → " + xp;
                        tr.appendChild(tdPrices);

                        const tdPnl = document.createElement("td");
                        const tag = document.createElement("span");
                        const cls = t.pnl_pct >= 0 ? "tag tag-win" : "tag tag-loss";
                        tag.className = cls;
                        tag.textContent = (t.pnl_pct * 100).toFixed(2) + "%";
                        tdPnl.appendChild(tag);
                        tr.appendChild(tdPnl);

                        const tdReason = document.createElement("td");
                        tdReason.textContent = t.reason || "";
                        tr.appendChild(tdReason);

                        tbody.appendChild(tr);
                    });

                    document.getElementById("last-update-trades").innerText =
                        "Оновлено: " + new Date().toLocaleString();
                } catch (e) {
                    console.error(e);
                }
            }

            // ===== Логи =====

            async function loadActions() {
                try {
                    const data = await fetchJSON("/api/actions?limit=50");
                    const tbody = document.getElementById("actions-body");
                    tbody.innerHTML = "";
                    data.actions.forEach(a => {
                        const tr = document.createElement("tr");

                        const tdTime = document.createElement("td");
                        tdTime.textContent = a.bar_time || "-";
                        tdTime.className = "small";
                        tr.appendChild(tdTime);

                        const tdSymbol = document.createElement("td");
                        tdSymbol.textContent = a.symbol || "-";
                        tr.appendChild(tdSymbol);

                        const tdAction = document.createElement("td");
                        tdAction.textContent = a.action || "";
                        tr.appendChild(tdAction);

                        const tdPrice = document.createElement("td");
                        if (a.price !== null && a.price !== undefined) {
                            tdPrice.textContent = a.price.toFixed(2);
                        } else {
                            tdPrice.textContent = "-";
                        }
                        tr.appendChild(tdPrice);

                        const tdPos = document.createElement("td");
                        tdPos.textContent = a.position_state || "";
                        tr.appendChild(tdPos);

                        const tdReason = document.createElement("td");
                        tdReason.textContent = a.reason || "";
                        tr.appendChild(tdReason);

                        tbody.appendChild(tr);
                    });

                    document.getElementById("last-update-actions").innerText =
                        "Оновлено: " + new Date().toLocaleString();
                } catch (e) {
                    console.error(e);
                }
            }

            function refreshAll() {
                loadSummary();
                loadOpenPositions();
                loadTrades();
                loadActions();
            }

            refreshAll();
            setInterval(refreshAll, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# =========================
# API ендпоінти
# =========================

@app.get("/api/summary")
def api_summary():
    try:
        df = load_trades()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trades_log.csv not found")
    summary = compute_summary(df)
    return JSONResponse(summary)


@app.get("/api/trades")
def api_trades(limit: int = 30):
    try:
        df = load_trades()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trades_log.csv not found")
    if df.empty:
        return {"trades": []}

    if "exit_time" in df.columns:
        df = df.sort_values("exit_time")
    df = df.tail(limit)

    trades = []
    for _, row in df.iterrows():
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time")

        if isinstance(entry_time, pd.Timestamp) and pd.notna(entry_time):
            entry_time_str = entry_time.isoformat()
        else:
            entry_time_str = str(entry_time) if entry_time is not None and not pd.isna(entry_time) else None

        if isinstance(exit_time, pd.Timestamp) and pd.notna(exit_time):
            exit_time_str = exit_time.isoformat()
        else:
            exit_time_str = str(exit_time) if exit_time is not None and not pd.isna(exit_time) else None

        pnl_pct_val = row.get("pnl_pct", 0)
        try:
            pnl_pct = float(pnl_pct_val) if pd.notna(pnl_pct_val) else 0.0
        except (TypeError, ValueError):
            pnl_pct = 0.0

        pnl_abs_val = row.get("pnl_abs", None)
        if pnl_abs_val is not None and not pd.isna(pnl_abs_val):
            try:
                pnl_abs = float(pnl_abs_val)
            except (TypeError, ValueError):
                pnl_abs = None
        else:
            pnl_abs = None

        trades.append({
            "symbol": row.get("symbol", ""),
            "entry_time": entry_time_str,
            "exit_time": exit_time_str,
            "direction": row.get("direction"),
            "entry_price": float(row.get("entry_price", 0)) if pd.notna(row.get("entry_price", 0)) else 0.0,
            "exit_price": float(row.get("exit_price", 0)) if pd.notna(row.get("exit_price", 0)) else 0.0,
            "pnl_pct": pnl_pct,
            "pnl_abs": pnl_abs,
            "reason": row.get("reason", ""),
        })
    return {"trades": trades}


@app.get("/api/actions")
def api_actions(limit: int = 50):
    try:
        df = load_actions()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="actions_log.csv not found")
    if df.empty:
        return {"actions": []}

    if "bar_time" in df.columns:
        df = df.sort_values("bar_time")
    df = df.tail(limit)

    actions = []
    for _, row in df.iterrows():
        bar_time = row.get("bar_time")
        if isinstance(bar_time, pd.Timestamp) and pd.notna(bar_time):
            bar_time_str = bar_time.isoformat()
        else:
            bar_time_str = str(bar_time) if bar_time is not None and not pd.isna(bar_time) else None

        price_val = row.get("price", None)
        if price_val is not None and not pd.isna(price_val):
            try:
                price = float(price_val)
            except (TypeError, ValueError):
                price = None
        else:
            price = None

        reason = row.get("diag_reason", None)
        if reason is None or (isinstance(reason, float) and pd.isna(reason)):
            reason = row.get("exit_reason", None)
        if reason is None or (isinstance(reason, float) and pd.isna(reason)):
            reason = row.get("entry_reason", "")
        if reason is None:
            reason = ""

        position_was_open = row.get("position_was_open", None)
        if isinstance(position_was_open, str):
            lowered = position_was_open.lower()
            if lowered in ("true", "1", "yes"):
                position_was_open = True
            elif lowered in ("false", "0", "no"):
                position_was_open = False

        if position_was_open is True:
            pos_state = "OPEN"
        elif position_was_open is False:
            pos_state = "FLAT"
        else:
            pos_state = ""

        actions.append({
            "bar_time": bar_time_str,
            "symbol": row.get("symbol", ""),
            "action": row.get("action", ""),
            "price": price,
            "position_state": pos_state,
            "reason": reason,
        })
    return {"actions": actions}


@app.get("/api/open_positions")
def api_open_positions():
    try:
        df_actions = load_actions()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="actions_log.csv not found")

    positions = compute_open_positions(df_actions)
    return {"positions": positions}
