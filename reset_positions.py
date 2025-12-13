import pandas as pd
from datetime import datetime, timezone

ACTIONS_FILE = "actions_log.csv"

# символи, які зараз "зависли" активними
SYMBOLS_TO_RESET = ["ETHUSDT", "TRXUSDT"]

df = pd.read_csv(ACTIONS_FILE)

now = datetime.now(timezone.utc).isoformat()

new_rows = []

for sym in SYMBOLS_TO_RESET:
    sub = df[df.get("symbol", "") == sym]
    if sub.empty:
        continue

    # беремо останній відомий price для цього символу
    last_row = sub.iloc[-1]
    last_price = last_row.get("price", "")

    row_dict = {}

    # проходимо по ВСІХ існуючих колонках, щоб структура не зламалась
    for col in df.columns:
        if col == "bar_time":
            row_dict[col] = now
        elif col == "symbol":
            row_dict[col] = sym
        elif col == "action":
            row_dict[col] = "CLOSE_RESET"  # починається з CLOSE_, дашборд сприйме як закриття
        elif col == "price":
            row_dict[col] = last_price
        elif col == "exit_reason":
            row_dict[col] = "manual_reset"
        elif col == "position_was_open":
            row_dict[col] = True
        else:
            # заповнюємо порожнім / NaN, щоб нічого не ламати
            row_dict[col] = ""

    new_rows.append(row_dict)

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(ACTIONS_FILE, index=False)
    print("Додано рядки для закриття:", SYMBOLS_TO_RESET)
else:
    print("Немає збігів по символам, нічого не змінено")
