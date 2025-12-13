from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
import subprocess
import time

app = FastAPI(title="OKHO Status")

SERVICES = [
    "okho-prod-bot", "okho-prod-dash",
    "okho-lowtp-bot", "okho-lowtp-dash",
    "okho-nodoge-bot", "okho-nodoge-dash",
    "okho-trail-bot", "okho-trail-dash",
]

def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def svc_status(name: str) -> dict:
    # "active", "inactive", "failed", etc.
    active = sh(["systemctl", "is-active", name])  # may raise on unknown
    # short human-friendly line
    info = sh(["systemctl", "show", name, "-p", "ActiveState", "-p", "SubState", "-p", "MemoryCurrent", "-p", "ExecMainStatus", "--no-pager"])
    d = {"name": name, "is_active": active == "active", "active": active}
    for line in info.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    # MemoryCurrent is bytes
    try:
        mem = int(d.get("MemoryCurrent", "0"))
        d["mem_mb"] = round(mem / (1024 * 1024), 1)
    except Exception:
        d["mem_mb"] = None
    return d

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OKHO status: /health (json), /text (short), /logs/{service} (tail)\n"

@app.get("/health")
def health():
    out = []
    ok = True
    for s in SERVICES:
        try:
            st = svc_status(s)
        except subprocess.CalledProcessError:
            st = {"name": s, "is_active": False, "active": "unknown_or_error"}
        out.append(st)
        if not st.get("is_active"):
            ok = False

    payload = {
        "ok": ok,
        "ts": int(time.time()),
        "services": out,
    }
    return JSONResponse(payload)

@app.get("/text", response_class=PlainTextResponse)
def text():
    lines = []
    ok = True
    for s in SERVICES:
        try:
            st = svc_status(s)
            status = "OK" if st["is_active"] else "DOWN"
            if not st["is_active"]:
                ok = False
            mem = st.get("mem_mb")
            mem_str = f"{mem}MB" if mem is not None else "?"
            lines.append(f"{status:4}  {s:18}  {mem_str}")
        except Exception:
            ok = False
            lines.append(f"DOWN  {s:18}  ?")
    head = "ALL OK ✅\n" if ok else "SOME DOWN ❌\n"
    return head + "\n".join(lines) + "\n"

@app.get("/logs/{service}", response_class=PlainTextResponse)
def logs(service: str, n: int = 120):
    if service not in SERVICES:
        return PlainTextResponse("Unknown service\n", status_code=404)
    try:
        out = sh(["journalctl", "-u", service, "-n", str(n), "--no-pager"])
        return out + "\n"
    except Exception as e:
        return PlainTextResponse(f"Error reading logs: {e}\n", status_code=500)
