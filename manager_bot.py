"""
╔══════════════════════════════════════════════════════════════╗
║            OllamaGate Manager Bot                            ║
║  Runs app.py via gunicorn + ngrok tunnel                     ║
║  and manages everything through Telegram.                    ║
╠══════════════════════════════════════════════════════════════╣
║  SETUP (one-time)                                            ║
║  ─────────────────                                           ║
║  1. Get a bot token from @BotFather on Telegram              ║
║     → /newbot  → copy the token                              ║
║                                                              ║
║  2. Get your Telegram user ID from @userinfobot              ║
║     (strongly recommended — restricts bot to only you)       ║
║                                                              ║
║  3. Export env vars:                                         ║
║       export TELEGRAM_BOT_TOKEN="123456:ABC..."              ║
║       export TELEGRAM_USER_ID="987654321"                    ║
║                                                              ║
║  4. Run from the same folder as app.py:                      ║
║       python3 manager_bot.py                                 ║
║                                                              ║
║  Dependencies are installed automatically on first run.      ║
║  Tunnel: ngrok http <port> — stable, streaming/SSE-capable.  ║
║  Requires: ngrok installed and authenticated on the host.    ║
╠══════════════════════════════════════════════════════════════╣
║  TELEGRAM COMMANDS                                           ║
║  ─────────────────                                           ║
║  /launch [duration]   Start server  (default 1h)             ║
║                       Examples: /launch 2h  /launch 30m      ║
║  /stop                Stop server + tunnel                   ║
║  /restart [duration]  Fresh token + restart                  ║
║  /status              Running status + time left             ║
║  /url                 Show tunnel URL                        ║
║  /token               Show OllamaGate access token           ║
║  /logs [n]            Last n log lines (default 30, max 100) ║
║  /models              List all Ollama + image models         ║
║  /load <model>        Pull/load a model into Ollama          ║
║  /chat <model>        Chat with a model directly in Telegram ║
║  /endchat             End the current chat session           ║
║  /clearchat           Reset chat conversation history        ║
║  /unload [model]      Unload model from VRAM (free memory)   ║
║  /clear               Clear log buffer + context             ║
║  /help                This command list                      ║
╠══════════════════════════════════════════════════════════════╣
║  STABILITY IMPROVEMENTS                                      ║
║  ──────────────────────                                      ║
║  • Tunnel watchdog: auto-reconnects if tunnel drops          ║
║    (e.g. after heavy image generation), sends Telegram alert ║
║  • ngrok URL fetched via local API (127.0.0.1:4040)          ║
║  • Reconnect restarts ngrok automatically                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess, sys

def _bootstrap():
    pkgs = {
        "telegram":  "python-telegram-bot>=20.0",
        "gunicorn":  "gunicorn",
        "gevent":    "gevent",
    }
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[setup] Installing: {', '.join(missing)} …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[setup] Done.")

_bootstrap()

import asyncio, json, os, platform, re, secrets, selectors, string
import textwrap, threading, time, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
import atexit

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
ALLOWED_USER = int(os.getenv("TELEGRAM_USER_ID", "0"))  #0 = no restriction (not recommended)
APP_DIR      = Path(__file__).parent
PORT         = int(os.getenv("PORT", "8000"))
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WSGI_FILE    = APP_DIR / "_gate_wsgi.py"
def _atexit_handler():
    """Clean up gunicorn and tunnel on process exit."""
    try:
        _watchdog_stop.set()
        with _lock:
            gun = _s.get("gun")
            tun = _s.get("tun")
        _kill(tun)
        _kill(gun)
    except Exception:
        pass

atexit.register(_atexit_handler)
_s = {
    "gun":  None,
    "tun":  None,
    "url":  None,
    "tok":  None,
    "exp":  None,
    "log":  [],
    "provider": None,
    "chat_model":   None,
    "chat_history": [],
}
_lock = threading.Lock()

_watchdog_thread: threading.Thread | None = None
_watchdog_stop   = threading.Event()
_app_ref: "Application | None" = None
_bot_loop: "asyncio.AbstractEventLoop | None" = None

def _now() -> datetime:
    return datetime.now(timezone.utc)

_WSGI_SRC = """\
import os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import (
    app, init_db,
    _state as gate_state, _state_lock,
    _expiry_watcher, _now,
)
from datetime import timedelta

init_db()

with _state_lock:
    gate_state["token"]      = os.environ["GATE_TOKEN"]
    gate_state["expires_at"] = _now() + timedelta(seconds=int(os.environ.get("GATE_SECS", "3600")))

threading.Thread(target=_expiry_watcher, daemon=True).start()
"""

def _make_token(n: int = 50) -> str:
    alpha = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alpha) for _ in range(n))


def _parse_dur(s: str) -> int | None:
    """'2h30m' → 9000 seconds.  Returns None on invalid input."""
    s = s.strip().lower().replace(" ", "")
    total = sum(
        int(val) * (3600 if unit == "h" else 60)
        for val, unit in re.findall(r"(\d+)([hm])", s)
    )
    return total or None


def _fmt_dur(secs: int) -> str:
    h, r = divmod(secs, 3600)
    m = r // 60
    return f"{h}h {m}m".strip() if h else f"{m}m"

def _tee(proc: subprocess.Popen, label: str):
    """Drain stdout + stderr of *proc* into the rolling log buffer."""
    sel = selectors.DefaultSelector()
    for fd in (proc.stdout, proc.stderr):
        if fd:
            sel.register(fd, selectors.EVENT_READ)
    while True:
        for key, _ in sel.select(timeout=0.5):
            raw = key.fileobj.readline()
            if raw:
                line = f"[{label}] {raw.decode(errors='replace').rstrip()}"
                with _lock:
                    _s["log"].append(line)
                    if len(_s["log"]) > 500:
                        _s["log"] = _s["log"][-400:]
        if proc.poll() is not None:
            break
    sel.close()

def _kill(proc: subprocess.Popen | None):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=6)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _health_check(url: str, retries: int = 5, delay: float = 3.0) -> bool:
    """
    Probe *url* up to *retries* times.  Any HTTP response (even 401/403) means
    the server is actually up and serving traffic.  Returns True on success.
    """
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "OllamaGate-HealthCheck/1.0")
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True
        except urllib.error.HTTPError:
            return True
        except Exception as exc:
            with _lock:
                _s["log"].append(f"[health] attempt {attempt+1}/{retries}: {exc}")
            if attempt < retries - 1:
                time.sleep(delay)
    return False

_NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def _ngrok_available() -> bool:
    try:
        subprocess.check_output(["ngrok", "version"], stderr=subprocess.STDOUT)
        return True
    except FileNotFoundError:
        return False

def _start_best_tunnel(port: int) -> tuple[subprocess.Popen | None, str | None, str | None]:
    """
    Start an ngrok tunnel on *port*.  Returns (proc, url, "ngrok")
    or (None, None, None) if ngrok is unavailable or fails to connect.
    URL is retrieved from the ngrok local REST API (127.0.0.1:4040).
    """
    if not _ngrok_available():
        with _lock:
            _s["log"].append("[tunnel] ngrok not found — install ngrok and authenticate first")
        return None, None, None

    with _lock:
        _s["log"].append(f"[tunnel] Starting ngrok http {port} …")

    proc = subprocess.Popen(
        ["ngrok", "http", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    threading.Thread(target=_tee, args=(proc, "ngrok"), daemon=True).start()

    deadline = time.time() + 30
    url = None
    while time.time() < deadline:
        if proc.poll() is not None:
            with _lock:
                _s["log"].append("[tunnel] ngrok process exited early — check auth/config")
            break
        try:
            with urllib.request.urlopen(_NGROK_API, timeout=3) as resp:
                data = json.loads(resp.read())
            for t in data.get("tunnels", []):
                if t.get("proto") == "https":
                    url = t["public_url"]
                    break
        except Exception:
            pass
        if url:
            break
        time.sleep(1)

    if url:
        with _lock:
            _s["log"].append(f"[tunnel] ✅ ngrok connected: {url}")
        return proc, url, "ngrok"

    _kill(proc)
    with _lock:
        _s["log"].append("[tunnel] ❌ ngrok failed to provide a URL — check /logs")
    return None, None, None

def _send_telegram_alert(text: str):
    """
    Fire-and-forget Telegram message from a background thread.
    Uses the stored _bot_loop (set during post_init) for thread-safe scheduling.
    """
    if _app_ref is None or ALLOWED_USER == 0 or _bot_loop is None:
        return

    async def _send():
        try:
            await _app_ref.bot.send_message(chat_id=ALLOWED_USER, text=text)
        except Exception as e:
            with _lock:
                _s["log"].append(f"[watchdog] telegram alert failed: {e}")

    try:
        asyncio.run_coroutine_threadsafe(_send(), _bot_loop)
    except Exception as e:
        with _lock:
            _s["log"].append(f"[watchdog] could not schedule alert: {e}")


def _watchdog_loop():
    """
    Runs in a daemon thread while the server is active.
    Checks every 15 s — if gunicorn is alive but tunnel is dead, reconnects.
    """
    POLL_INTERVAL = 15

    while not _watchdog_stop.is_set():
        _watchdog_stop.wait(POLL_INTERVAL)
        if _watchdog_stop.is_set():
            break

        with _lock:
            gun_alive = bool(_s["gun"] and _s["gun"].poll() is None)
            tun_alive = bool(_s["tun"] and _s["tun"].poll() is None)

        if not gun_alive:
            continue

        if tun_alive:
            continue

        with _lock:
            _s["log"].append("[watchdog] ⚠️  Tunnel died — attempting auto-reconnect…")

        _send_telegram_alert(
            "⚠️ OllamaGate: tunnel dropped — reconnecting automatically…"
        )

        tun, url, provider = _start_best_tunnel(PORT)

        with _lock:
            _s["tun"] = tun
            _s["url"] = url
            _s["provider"] = provider

        if url:
            with _lock:
                _s["log"].append(f"[watchdog] ✅ Reconnected via {provider}: {url}")
            _send_telegram_alert(
                f"✅ OllamaGate: tunnel reconnected!\n\n"
                f"🌐 New URL:\n{url}\n\n"
                f"🔑 Token unchanged — use /token to see it.\n"
                f"📡 Provider: {provider}"
            )
        else:
            with _lock:
                _s["log"].append("[watchdog] ❌ Auto-reconnect failed — all providers down.")
            _send_telegram_alert(
                "❌ OllamaGate: tunnel reconnect FAILED.\n"
                "All providers unreachable. Use /restart to try again."
            )


def _start_watchdog():
    global _watchdog_thread
    _watchdog_stop.clear()
    _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True, name="tunnel-watchdog")
    _watchdog_thread.start()


def _stop_watchdog():
    _watchdog_stop.set()


def do_launch(tok: str, secs: int) -> tuple[str | None, bool]:
    """
    Blocking — writes WSGI wrapper, starts gunicorn + tunnel, returns (url, health_ok).
    """
    WSGI_FILE.write_text(_WSGI_SRC)
    dur_str = f"{secs // 3600}h" if secs % 3600 == 0 else f"{secs // 60}m"
    env = {**os.environ, "GATE_TOKEN": tok, "GATE_SECS": str(secs), "DURATION": dur_str}

    gun = subprocess.Popen(
        [
            sys.executable, "-m", "gunicorn",
            "-k", "gevent",
            "-w", "1",
            "-b", f"0.0.0.0:{PORT}",
            "--timeout", "180",
            "--keep-alive", "30",
            "--worker-connections", "100",
            "--forwarded-allow-ips", "*",
            "--access-logfile", "-",
            "--error-logfile",  "-",
            "_gate_wsgi:app",
        ],
        cwd=str(APP_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    threading.Thread(target=_tee, args=(gun, "gunicorn"), daemon=True).start()
    time.sleep(8)

    local_url = f"http://127.0.0.1:{PORT}"
    with _lock:
        _s["log"].append(f"[health] Checking local server at {local_url} …")
    local_ok = _health_check(local_url, retries=6, delay=2.0)
    if not local_ok:
        with _lock:
            _s["log"].append(
                "[health] ❌ Local server did not respond — gunicorn failed to start. "
                "Check /logs for errors (port in use? app.py import error?)."
            )
        _kill(gun)
        return None, False
    with _lock:
        _s["log"].append(f"[health] ✅ Local server is up on port {PORT}")

    tun, url, provider = _start_best_tunnel(PORT)

    health_ok = False
    if url:
        with _lock:
            _s["log"].append(f"[health] Probing tunnel URL {url} …")
        health_ok = _health_check(url, retries=8, delay=3.0)
        with _lock:
            _s["log"].append(
                f"[health] {'✅ tunnel is responding' if health_ok else '❌ tunnel did NOT respond — check /logs'}"
            )

    with _lock:
        _s.update(gun=gun, tun=tun, url=url, tok=tok,
                  exp=_now() + timedelta(seconds=secs),
                  provider=provider)

    _start_watchdog()

    return url, health_ok


def do_stop():
    """Blocking — terminates gunicorn + tunnel, clears state."""
    _stop_watchdog()
    with _lock:
        gun, tun = _s["gun"], _s["tun"]
        _s.update(gun=None, tun=None, url=None, tok=None, exp=None, provider=None)
    _kill(tun)
    _kill(gun)

def _is_running() -> bool:
    with _lock:
        return bool(_s["gun"] and _s["gun"].poll() is None)

def _e(s: str) -> str:
    """Escape a plain string for Telegram MarkdownV2."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        s = s.replace(ch, f"\\{ch}")
    return s

def _allowed(update: Update) -> bool:
    if ALLOWED_USER == 0:
        return True
    return update.effective_user.id == ALLOWED_USER

def _ollama_list_models() -> list[dict] | None:
    """
    Query Ollama's local REST API for installed models.
    Returns list of model dicts, or None if Ollama is unreachable.
    """
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            headers={"User-Agent": "OllamaGate-Manager/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data.get("models", [])
    except Exception:
        return None


def _ollama_pull(model: str, log_prefix: str = "pull") -> subprocess.Popen:
    """Start 'ollama pull <model>' and tee its output."""
    proc = subprocess.Popen(
        ["ollama", "pull", model],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    threading.Thread(target=_tee, args=(proc, log_prefix), daemon=True).start()
    return proc

_IMAGE_MODEL_KEYWORDS = [
    "stable-diffusion", "stablediffusion", "sdxl", "sd3", "flux",
    "dall-e", "dalle", "dreamshaper", "realisticvision", "animagine",
    "playgroundv", "wuerstchen", "deepfloyd", "kandinsky",
]

def _is_image_model(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _IMAGE_MODEL_KEYWORDS)


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    return f"{size_bytes / 1_000_000:.0f} MB"

async def cmd_help(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return
    await u.message.reply_text(
        "🤖 *OllamaGate Manager Bot*\n\n"
        "/launch \\[duration\\] — start server\n"
        "    _e\\.g\\._ `/launch 2h` `/launch 30m` `/launch 1h30m`\n\n"
        "/stop — stop server \\+ tunnel\n\n"
        "/restart \\[duration\\] — stop \\+ fresh token \\+ restart\n\n"
        "/status — show running status \\+ time remaining\n\n"
        "/url — show the public tunnel URL\n\n"
        "/token — show the OllamaGate access token\n\n"
        "/logs \\[n\\] — last n log lines \\(default 30, max 100\\)\n\n"
        "/models — list all installed Ollama \\+ image models\n\n"
        "/load \\<model\\> — pull/load a model into Ollama\n"
        "    _e\\.g\\._ `/load llama3:8b` `/load mistral`\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "💬 *Direct Chat*\n\n"
        "/chat \\<model\\> — start chatting with a model directly in Telegram\n"
        "    _e\\.g\\._ `/chat llama3:8b` — then just type messages freely\n\n"
        "/endchat — end the current chat session\n\n"
        "/clearchat — reset conversation history \\(keep session open\\)\n\n"
        "/unload \\[model\\] — unload model from VRAM to free memory\n"
        "    _Omit model name to unload the active chat model_\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "/shutdown confirm — shut down the host PC \\(stops server first\\)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "/clear — clear log buffer and reset context\n\n"
        "/help — show this message\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🛡 *Tunnel auto\\-reconnects* if it drops \\(e\\.g\\. after image gen\\)\\.\n"
        "You'll get a Telegram alert with the new URL automatically\\.",
        parse_mode="MarkdownV2",
    )


async def cmd_launch(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    if _is_running():
        await u.message.reply_text(
            "⚠️ Server is already running\\.\n"
            "Use /stop first, or /restart to get a fresh token\\.",
            parse_mode="MarkdownV2",
        )
        return

    raw  = " ".join(c.args) if c.args else "1h"
    secs = _parse_dur(raw)
    if not secs:
        await u.message.reply_text(
            f"❌ Invalid duration `{_e(raw)}`\\.\n"
            "Try `/launch 2h` or `/launch 30m`\\.",
            parse_mode="MarkdownV2",
        )
        return

    tok = _make_token()
    await u.message.reply_text(
        f"🚀 Launching server for *{_e(raw)}*…\n"
        "_This takes about 20–30 seconds\\._",
        parse_mode="MarkdownV2",
    )

    url, health_ok = await asyncio.to_thread(do_launch, tok, secs)

    if url and health_ok:
        msg = (
            "✅ *Server is live and responding\\!*\n\n"
            f"🌐 *Tunnel URL*\n`{_e(url)}`\n\n"
            f"🔑 *Access Token*\n`{_e(tok)}`\n\n"
            f"⏰ *Session expires in:* {_e(_fmt_dur(secs))}\n\n"
            "🛡 _Tunnel watchdog is active — auto\\-reconnects if tunnel drops\\._"
        )
    elif url and not health_ok:
        msg = (
            "⚠️ *Tunnel URL found but did not respond yet\\.*\n\n"
            f"🌐 URL: `{_e(url)}`\n\n"
            f"🔑 *Token:* `{_e(tok)}`\n\n"
            "The tunnel may need a few more seconds\\. Try the URL in your browser\\.\n"
            "If it still fails after 30 s, use /logs to check for errors\\."
        )
    elif not url:
        msg = (
            "❌ *Gunicorn failed to start on port " + str(PORT) + "\\.*\n\n"
            "Possible causes:\n"
            "• Another process is already using port " + str(PORT) + "\n"
            "• app\\.py has an import error\n\n"
            "Run `/logs 50` to see gunicorn's error output\\."
        )

    elif not url:
        msg = (
            "❌ *Gunicorn or tunnel failed\\.*\n\n"
            "• If gunicorn failed: another process may be on port " + str(PORT) + ", "
            "or app\\.py has an import error\\.\n"
            "• If tunnel failed: all SSH tunnel providers were unreachable\\.\n\n"
            "Run `/logs 50` to see the full error output\\."
        )

    await u.message.reply_text(msg, parse_mode="MarkdownV2")


async def cmd_stop(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    if not _is_running():
        await u.message.reply_text("ℹ️ Server is not running\\.", parse_mode="MarkdownV2")
        return

    await u.message.reply_text("🛑 Stopping server and tunnel…")
    await asyncio.to_thread(do_stop)
    await u.message.reply_text("✅ Server stopped\\.", parse_mode="MarkdownV2")


async def cmd_restart(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    raw  = " ".join(c.args) if c.args else "1h"
    secs = _parse_dur(raw)
    if not secs:
        await u.message.reply_text(
            f"❌ Invalid duration `{_e(raw)}`\\. Try `/restart 2h`\\.",
            parse_mode="MarkdownV2",
        )
        return

    await u.message.reply_text("🔄 Restarting — stopping current instance…")
    await asyncio.to_thread(do_stop)
    await asyncio.sleep(2)

    tok = _make_token()
    await u.message.reply_text(
        f"🚀 Starting fresh for *{_e(raw)}*…",
        parse_mode="MarkdownV2",
    )
    url, health_ok = await asyncio.to_thread(do_launch, tok, secs)

    if url and health_ok:
        msg = (
            "✅ *Restarted successfully\\!*\n\n"
            f"🌐 *Tunnel URL*\n`{_e(url)}`\n\n"
            f"🔑 *Access Token*\n`{_e(tok)}`\n\n"
            f"⏰ *Session expires in:* {_e(_fmt_dur(secs))}\n\n"
            "🛡 _Tunnel watchdog is active\\._"
        )
    elif url and not health_ok:
        msg = (
            "⚠️ Restarted — tunnel URL found but not yet responding\\.\n\n"
            f"🌐 URL: `{_e(url)}`\n\n"
            f"🔑 *Token:* `{_e(tok)}`\n\n"
            "Try the URL in your browser\\. If it fails after 30 s, check /logs\\."
        )
    else:
        msg = (
            "❌ *Restart failed — gunicorn or tunnel did not start\\.*\n\n"
            f"🔑 *Token:* `{_e(tok)}`\n\n"
            "Run `/logs 50` to see what went wrong\\."
        )
    await u.message.reply_text(msg, parse_mode="MarkdownV2")


async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    with _lock:
        gun_ok   = bool(_s["gun"] and _s["gun"].poll() is None)
        tun_ok   = bool(_s["tun"] and _s["tun"].poll() is None)
        url      = _s["url"]
        exp      = _s["exp"]
        tok_set  = bool(_s["tok"])
        provider = _s["provider"] or "unknown"

    if not gun_ok:
        await u.message.reply_text("🔴 Server is *not running*\\.", parse_mode="MarkdownV2")
        return

    time_left = ""
    if exp:
        secs_left = max(0, int((exp - _now()).total_seconds()))
        time_left = _fmt_dur(secs_left) if secs_left else "expired"

    lines = [
        "🟢 *Server is running*\n",
        f"🌐 URL: `{_e(url)}`" if url else "🌐 URL: _not yet captured_",
        f"⏰ Time left: {_e(time_left)}" if time_left else "⏰ Time left: _unknown_",
        f"📡 Gunicorn: {'✅ alive' if gun_ok else '❌ dead'}",
        f"🔗 Tunnel:   {'✅ alive' if tun_ok else '❌ dead \\(watchdog reconnecting…\\)'}",
        f"🚇 Provider: {_e(provider)}",
        f"🔑 Token:    {'✅ set' if tok_set else '❌ missing'}",
        f"🛡 Watchdog: {'✅ active' if not _watchdog_stop.is_set() else '❌ stopped'}",
    ]
    await u.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")


async def cmd_url(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    if not _is_running():
        await u.message.reply_text("🔴 Server is not running\\.", parse_mode="MarkdownV2")
        return

    with _lock:
        url = _s["url"]
        if not url:
            blob = " ".join(_s["log"][-60:])
            for _, _, url_re, _ in TUNNEL_PROVIDERS:
                m = url_re.search(blob)
                if m:
                    url = m.group(0)
                    _s["url"] = url
                    break

    if url:
        await u.message.reply_text(
            f"🌐 *Tunnel URL:*\n`{_e(url)}`",
            parse_mode="MarkdownV2",
        )
    else:
        await u.message.reply_text(
            "❓ URL not captured yet\\. The watchdog will update it when the tunnel connects\\.\n"
            "Check /logs for tunnel output\\.",
            parse_mode="MarkdownV2",
        )


async def cmd_token(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    if not _is_running():
        await u.message.reply_text("🔴 Server is not running\\.", parse_mode="MarkdownV2")
        return

    with _lock:
        tok = _s["tok"]
        exp = _s["exp"]

    if tok:
        exp_str = exp.strftime("%H:%M UTC") if exp else "unknown"
        await u.message.reply_text(
            f"🔑 *OllamaGate Access Token*\n\n"
            f"`{_e(tok)}`\n\n"
            f"⏰ Valid until: {_e(exp_str)}",
            parse_mode="MarkdownV2",
        )
    else:
        await u.message.reply_text("❓ Token not set\\.", parse_mode="MarkdownV2")


async def cmd_logs(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not _allowed(u): return

    n = 30
    if c.args:
        try:
            n = min(max(int(c.args[0]), 1), 100)
        except ValueError:
            pass

    with _lock:
        lines = list(_s["log"][-n:])

    if not lines:
        await u.message.reply_text("📋 No logs yet\\.", parse_mode="MarkdownV2")
        return

    text = "\n".join(lines)
    if len(text) > 3800:
        text = "…(truncated)\n" + text[-3800:]
    text = text.replace("`", "'")

    await u.message.reply_text(
        f"```\n{text}\n```",
        parse_mode="MarkdownV2",
    )


async def cmd_models(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /models — list all models known to Ollama (LLM + image), grouped by type.
    """
    if not _allowed(u): return

    await u.message.reply_text("🔍 Querying Ollama for installed models…")

    models = await asyncio.to_thread(_ollama_list_models)

    if models is None:
        await u.message.reply_text(
            "❌ *Could not reach Ollama*\\.\n\n"
            f"Is Ollama running at `{_e(OLLAMA_HOST)}`?\n"
            "Make sure Ollama is started before using this command\\.",
            parse_mode="MarkdownV2",
        )
        return

    if not models:
        await u.message.reply_text(
            "📭 No models installed yet\\.\n\n"
            f"Use `/load <model>` to pull one, e\\.g\\. `/load llama3:8b`",
            parse_mode="MarkdownV2",
        )
        return

    llm_lines   = []
    image_lines = []

    for m in sorted(models, key=lambda x: x.get("name", "")):
        name = m.get("name", "?")
        size = _fmt_size(m.get("size", 0))
        modified = m.get("modified_at", "")[:10] if m.get("modified_at") else ""
        entry = f"  • `{_e(name)}` — {_e(size)}" + (f" \\({_e(modified)}\\)" if modified else "")
        if _is_image_model(name):
            image_lines.append(entry)
        else:
            llm_lines.append(entry)

    parts = [f"📦 *Installed Models* \\({len(models)} total\\)\n"]

    if llm_lines:
        parts.append("🧠 *LLM / Text models:*")
        parts.extend(llm_lines)

    if image_lines:
        parts.append("\n🎨 *Image models:*")
        parts.extend(image_lines)

    parts.append(
        f"\n_Use `/load <name>` to add more\\._\n"
        f"_Ollama host: `{_e(OLLAMA_HOST)}`_"
    )

    await u.message.reply_text("\n".join(parts), parse_mode="MarkdownV2")


async def cmd_load(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /load <model> — pull (download + load) a model into Ollama.
    Example: /load llama3:8b   /load mistral   /load nomic-embed-text
    """
    if not _allowed(u): return

    if not c.args:
        await u.message.reply_text(
            "❌ Please specify a model name\\.\n\n"
            "*Usage:* `/load <model>`\n"
            "*Examples:*\n"
            "  `/load llama3:8b`\n"
            "  `/load mistral`\n"
            "  `/load nomic\\-embed\\-text`\n\n"
            "See available models at: https://ollama\\.com/library",
            parse_mode="MarkdownV2",
        )
        return

    model = c.args[0].strip()
    safe_model = _e(model)

    await u.message.reply_text(
        f"⬇️ Pulling model `{safe_model}`…\n\n"
        "_This may take several minutes depending on model size\\._\n"
        "_Progress is logged — use /logs to monitor\\._",
        parse_mode="MarkdownV2",
    )

    def _do_pull():
        with _lock:
            _s["log"].append(f"[load] Starting: ollama pull {model}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True, text=True, timeout=1800,
            )
            with _lock:
                for line in (result.stdout + result.stderr).splitlines():
                    _s["log"].append(f"[load] {line}")
            return result.returncode == 0, result.stderr
        except FileNotFoundError:
            msg = "ollama command not found — is Ollama installed?"
            with _lock:
                _s["log"].append(f"[load] ❌ {msg}")
            return False, msg
        except subprocess.TimeoutExpired:
            msg = "Pull timed out after 30 minutes."
            with _lock:
                _s["log"].append(f"[load] ❌ {msg}")
            return False, msg
        except Exception as e:
            with _lock:
                _s["log"].append(f"[load] ❌ {e}")
            return False, str(e)

    success, err = await asyncio.to_thread(_do_pull)

    if success:
        await u.message.reply_text(
            f"✅ *Model `{safe_model}` loaded successfully\\!*\n\n"
            "Use /models to see all installed models\\.",
            parse_mode="MarkdownV2",
        )
    else:
        short_err = _e((err or "unknown error")[:300])
        await u.message.reply_text(
            f"❌ *Failed to load `{safe_model}`*\n\n"
            f"`{short_err}`\n\n"
            "Check /logs for full output\\.",
            parse_mode="MarkdownV2",
        )


async def cmd_clear(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /clear — wipe the rolling log buffer and reset conversation context.
    Useful to free memory and start a clean session without restarting.
    """
    if not _allowed(u): return

    with _lock:
        log_count = len(_s["log"])
        _s["log"].clear()
        _s["log"].append(f"[clear] Log cleared by user at {_now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    await u.message.reply_text(
        f"🧹 *Cleared\\!*\n\n"
        f"📋 Log buffer: removed {_e(str(log_count))} lines\n"
        f"🧠 Context: reset\n\n"
        "_Server and tunnel remain running\\._\n"
        "_Use /status to check, /logs to see fresh output\\._",
        parse_mode="MarkdownV2",
    )


async def cmd_chat(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /chat <model> — start an interactive chat session with an Ollama model.
    All plain text messages will be forwarded to the model while chat is active.
    """
    if not _allowed(u): return

    if not c.args:
        with _lock:
            current = _s["chat_model"]
        if current:
            await u.message.reply_text(
                f"💬 Currently chatting with `{_e(current)}`\\.\n\n"
                "Just send messages to talk\\. Commands:\n"
                "/endchat — end the session\n"
                "/clearchat — reset conversation history",
                parse_mode="MarkdownV2",
            )
        else:
            await u.message.reply_text(
                "❌ No model specified\\.\n\n"
                "*Usage:* `/chat <model>`\n"
                "*Example:* `/chat llama3:8b`\n\n"
                "Use /models to see installed models\\.",
                parse_mode="MarkdownV2",
            )
        return

    model = c.args[0].strip()
    with _lock:
        _s["chat_model"] = model
        _s["chat_history"] = []

    await u.message.reply_text(
        f"💬 *Chat started with* `{_e(model)}`\n\n"
        "Just type your messages to talk with the AI\\.\n\n"
        "• /endchat — end the session\n"
        "• /clearchat — reset conversation history\n"
        "• /unload — unload model from VRAM when done",
        parse_mode="MarkdownV2",
    )


async def cmd_endchat(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /endchat — stop the active Ollama chat session.
    """
    if not _allowed(u): return

    with _lock:
        model = _s["chat_model"]
        _s["chat_model"] = None
        _s["chat_history"] = []

    if model:
        await u.message.reply_text(
            f"👋 Chat with `{_e(model)}` ended\\.\n\n"
            f"_Use `/unload {_e(model)}` to free VRAM, or /chat to start a new session\\._",
            parse_mode="MarkdownV2",
        )
    else:
        await u.message.reply_text("ℹ️ No active chat session\\.", parse_mode="MarkdownV2")


async def cmd_clearchat(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /clearchat — reset conversation history while keeping the chat session open.
    """
    if not _allowed(u): return

    with _lock:
        model = _s["chat_model"]
        _s["chat_history"] = []

    if model:
        await u.message.reply_text(
            f"🧹 Conversation history cleared\\.\n"
            f"Still chatting with `{_e(model)}`\\.",
            parse_mode="MarkdownV2",
        )
    else:
        await u.message.reply_text("ℹ️ No active chat session\\.", parse_mode="MarkdownV2")


async def cmd_unload(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /unload [model] — unload a model from VRAM without deleting it.
    If no model is given, unloads the currently active chat model.
    """
    if not _allowed(u): return

    if c.args:
        model = c.args[0].strip()
    else:
        with _lock:
            model = _s["chat_model"]
        if not model:
            await u.message.reply_text(
                "❌ No model specified and no active chat session\\.\n\n"
                "*Usage:* `/unload <model>`\n"
                "*Example:* `/unload llama3:8b`",
                parse_mode="MarkdownV2",
            )
            return

    safe_model = _e(model)
    await u.message.reply_text(
        f"⏏️ Unloading `{safe_model}` from VRAM…",
        parse_mode="MarkdownV2",
    )

    def _do_unload():
        try:
            payload = json.dumps({"model": model, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "OllamaGate-Manager/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return True, None
        except Exception as exc:
            return False, str(exc)

    success, err = await asyncio.to_thread(_do_unload)

    with _lock:
        if _s["chat_model"] == model:
            _s["chat_model"] = None
            _s["chat_history"] = []

    if success:
        await u.message.reply_text(
            f"✅ `{safe_model}` unloaded from VRAM\\.\n\n"
            f"_The model is still installed — use `/chat {safe_model}` to reload it\\._",
            parse_mode="MarkdownV2",
        )
    else:
        short_err = _e((err or "unknown error")[:300])
        await u.message.reply_text(
            f"❌ Failed to unload `{safe_model}`\\:\n\n`{short_err}`\n\n"
            f"Is Ollama running at `{_e(OLLAMA_HOST)}`?",
            parse_mode="MarkdownV2",
        )


async def cmd_shutdown(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    /shutdown — stop the server/tunnel cleanly, then shut down the host PC.
    A confirmation step is required to prevent accidental shutdowns.
    """
    if not _allowed(u): return

    args = c.args
    if not args or args[0].lower() != "confirm":
        await u.message.reply_text(
            "⚠️ *This will shut down the PC running this bot\\!*\n\n"
            "To confirm, send:\n"
            "`/shutdown confirm`\n\n"
            "_The server and tunnel will be stopped first\\._",
            parse_mode="MarkdownV2",
        )
        return

    await u.message.reply_text(
        "🛑 Stopping server and tunnel…\n"
        "💤 PC will shut down in 5 seconds\\. Goodbye\\!",
        parse_mode="MarkdownV2",
    )

    await asyncio.to_thread(do_stop)

    def _do_shutdown():
        time.sleep(5)
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=False)
        else:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    threading.Thread(target=_do_shutdown, daemon=True).start()


async def on_message(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """
    Handles all plain text messages.
    When a /chat session is active, forwards the message to Ollama and replies.
    """
    if not _allowed(u): return
    if not u.message or not u.message.text:
        return

    with _lock:
        model = _s["chat_model"]
        history = list(_s["chat_history"])

    if not model:
        return

    user_text = u.message.text.strip()
    if not user_text:
        return

    await u.message.chat.send_action("typing")

    history.append({"role": "user", "content": user_text})

    def _do_chat():
        try:
            payload = json.dumps({
                "model": model,
                "messages": history,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/chat",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "OllamaGate-Manager/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                return data.get("message", {}).get("content", ""), None
        except Exception as exc:
            return None, str(exc)

    reply_text, err = await asyncio.to_thread(_do_chat)

    if reply_text is not None:
        history.append({"role": "assistant", "content": reply_text})
        with _lock:
            if _s["chat_model"] == model:
                _s["chat_history"] = history

        MAX_LEN = 4000
        chunks = [reply_text[i:i + MAX_LEN] for i in range(0, len(reply_text), MAX_LEN)]
        for chunk in chunks:
            await u.message.reply_text(chunk)
    else:
        short_err = (err or "unknown error")[:300]
        await u.message.reply_text(
            f"❌ Ollama error: {short_err}\n\n"
            f"Is `{model}` installed? Use /models to check, or /endchat to exit."
        )

_COMMANDS = [
    ("launch",    "Start server — /launch 2h"),
    ("stop",      "Stop server + tunnel"),
    ("restart",   "Fresh restart — /restart 1h"),
    ("status",    "Show running status"),
    ("url",       "Show tunnel URL"),
    ("token",     "Show access token"),
    ("logs",      "Show recent logs — /logs 50"),
    ("models",    "List all installed Ollama + image models"),
    ("load",      "Pull/load a model — /load llama3:8b"),
    ("chat",      "Chat with a model — /chat llama3:8b"),
    ("endchat",   "End the current chat session"),
    ("clearchat", "Reset chat conversation history"),
    ("unload",    "Unload model from VRAM — /unload llama3:8b"),
    ("shutdown",  "Shut down the host PC — /shutdown confirm"),
    ("clear",     "Clear log buffer and context"),
    ("help",      "Show command list"),
]

def main():
    global _app_ref

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print(
            "\n❌  TELEGRAM_BOT_TOKEN is not set.\n"
            "    export TELEGRAM_BOT_TOKEN='123456:ABC-...'\n"
            "    Get one from @BotFather on Telegram.\n"
        )
        sys.exit(1)

    if not (APP_DIR / "app.py").exists():
        print(f"\n❌  app.py not found in {APP_DIR}\n"
              "    Run manager_bot.py from the same folder as app.py.\n")
        sys.exit(1)

    if ALLOWED_USER == 0:
        print(
            "\n⚠️  TELEGRAM_USER_ID is not set — the bot will respond to ANY user.\n"
            "    Set it to your user ID for security:\n"
            "    export TELEGRAM_USER_ID='987654321'\n"
            "    (Get your ID from @userinfobot on Telegram)\n"
        )

    print("🤖 Starting OllamaGate Manager Bot …")
    print(f"   App dir  : {APP_DIR}")
    print(f"   Port     : {PORT}")
    print(f"   Ollama   : {OLLAMA_HOST}")
    print(f"   Allowed  : {ALLOWED_USER or 'everyone (no restriction)'}")
    print()

    _http = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    application = Application.builder().token(BOT_TOKEN).request(_http).build()
    _app_ref = application

    handlers = [
        ("start",     cmd_help),
        ("help",      cmd_help),
        ("launch",    cmd_launch),
        ("stop",      cmd_stop),
        ("restart",   cmd_restart),
        ("status",    cmd_status),
        ("url",       cmd_url),
        ("token",     cmd_token),
        ("logs",      cmd_logs),
        ("models",    cmd_models),
        ("load",      cmd_load),
        ("chat",      cmd_chat),
        ("endchat",   cmd_endchat),
        ("clearchat", cmd_clearchat),
        ("unload",    cmd_unload),
        ("shutdown",  cmd_shutdown),
        ("clear",     cmd_clear),
    ]
    for name, fn in handlers:
        application.add_handler(CommandHandler(name, fn))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    async def _post_init(app):
        global _bot_loop
        _bot_loop = asyncio.get_event_loop()
        await app.bot.set_my_commands(
            [BotCommand(cmd, desc) for cmd, desc in _COMMANDS]
        )

    application.post_init = _post_init

    print("✅ Bot is running. Open Telegram and send /help to your bot.")
    print("   Press Ctrl+C to stop.\n")
    application.run_polling(drop_pending_updates=True, bootstrap_retries=-1)


if __name__ == "__main__":
    main()