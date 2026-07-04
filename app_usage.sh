#!/usr/bin/env bash
#
# Live resource usage for THIS app only — the `streamlit run app.py` process plus its
# descendants (the in-process Surya model, and the llama-server OCR backend it spawns) —
# with the two generation speeds that matter:
#
#   • Surya OCR  tokens/sec — polled live from llama-server's /slots (n_decoded deltas).
#   • Local LM   tokens/sec — read from .llm_usage.json, which the app writes after each
#                             extraction call. LM Studio (MLX) / Ollama expose no live
#                             tok/s endpoint, so this per-call record is the only source.
#
#   ./app_usage.sh          # live, refresh every 1s (Ctrl-C to stop)
#   ./app_usage.sh 2        # refresh every 2s
#   ./app_usage.sh once     # a single snapshot, then exit
#
# Flags a red headroom warning when free system RAM drops below MRIE_MEM_ALARM_GB
# (default 4) — i.e. when the app is close to forcing swap.

HERE="$(cd "$(dirname "$0")" && pwd)"
export MRIE_APP_DIR="$HERE"

exec python3 - "${1:-1}" <<'PY'
import os, sys, re, time, json, subprocess, urllib.request

arg = sys.argv[1] if len(sys.argv) > 1 else ""
once = arg == "once"
# A tok/s rate needs two samples over time, so `once` warms up a few quick samples
# (building the OCR decode history) and prints only the last one.
interval = 0.5 if once else (float(arg) if arg else 1.0)
WARMUP = 4
tty = sys.stdout.isatty()
alarm_gb = float(os.environ.get("MRIE_MEM_ALARM_GB", "4"))
RED, DIM, RST = ("\033[31m", "\033[2m", "\033[0m") if tty else ("", "", "")
GB = 1024 ** 3
STATS_PATH = os.path.join(os.environ.get("MRIE_APP_DIR", "."), ".llm_usage.json")
OCR_PATH = os.path.join(os.environ.get("MRIE_APP_DIR", "."), ".ocr_usage.json")


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def free_ram_bytes(memsize: int, pagesize: int) -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {}
    for line in out.splitlines():
        m = re.match(r'"?(.+?)"?:\s+(\d+)\.', line)
        if m:
            pages[m.group(1).strip()] = int(m.group(2))
    used = (pages.get("Pages active", 0) + pages.get("Pages wired down", 0)
            + pages.get("Pages occupied by compressor", 0)) * pagesize
    return memsize - used


def is_app_root(cmd: str) -> bool:
    low = cmd.lower()
    return ("streamlit" in low and "app.py" in low) or ("llama-server" in low and "surya" in low)


def label(cmd: str) -> str:
    low = cmd.lower()
    if "llama-server" in low:
        return "llama-server (Surya OCR)"
    if "streamlit" in low and "app.py" in low:
        return "streamlit (app + Surya)"
    tok = cmd.split()[0] if cmd.split() else cmd
    return tok.split("/")[-1]


def app_processes():
    """The app root(s) plus every descendant, as [(rss_kb, cpu, cmd, pid), …]."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,%cpu=,command="],
                         capture_output=True, text=True).stdout
    procs, children = {}, {}
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5 or not (parts[0].isdigit() and parts[2].isdigit()):
            continue
        pid, ppid, rss, cpu, cmd = parts
        procs[pid] = {"ppid": ppid, "rss": int(rss), "cpu": float(cpu), "cmd": cmd}
        children.setdefault(ppid, []).append(pid)
    seen, stack = set(), [pid for pid, i in procs.items() if is_app_root(i["cmd"])]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    rows = [(procs[p]["rss"], procs[p]["cpu"], procs[p]["cmd"], p) for p in seen]
    rows.sort(reverse=True)
    return rows


def llama_port(pid: str):
    lsof = subprocess.run(["lsof", "-nP", "-p", pid], capture_output=True, text=True).stdout
    m = re.search(r"127\.0\.0\.1:(\d+)", lsof)
    return m.group(1) if m else None


def surya_slots(port: str):
    try:
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=2).read())
    except Exception:
        return None


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# Persistent state across refreshes: per-slot decoded counts (for OCR tok/s deltas),
# a short rolling window, and a cached llama-server port so we don't lsof every tick.
prev, history, port_cache = {}, [], {"pid": None, "port": None}

try:
    memsize = int(sysctl("hw.memsize"))
    pagesize = int(sysctl("hw.pagesize")) or 16384
    ncpu = sysctl("hw.ncpu")
    iters = 0
    while True:
        rows = app_processes()
        rss_gb = sum(r for r, _, _, _ in rows) / 1048576
        cpu = sum(c for _, c, _, _ in rows)
        swap = re.findall(r"([\d.]+)M", sysctl("vm.swapusage"))
        swap_used = float(swap[1]) / 1024 if len(swap) > 1 else 0

        # --- Surya OCR tokens/sec (live from llama-server /slots) ---
        llama_pid = next((p for _, _, cmd, p in rows if "llama-server" in cmd.lower()), None)
        surya_line = "llama-server not running (spawns on first OCR)"
        if llama_pid:
            if port_cache["pid"] != llama_pid or not port_cache["port"]:
                port_cache.update(pid=llama_pid, port=llama_port(llama_pid))
            slots = surya_slots(port_cache["port"]) if port_cache["port"] else None
            if slots is None:  # port may have changed; re-resolve once
                port_cache["port"] = llama_port(llama_pid)
                slots = surya_slots(port_cache["port"]) if port_cache["port"] else None
            if slots is not None:
                now = time.time()
                delta = 0
                for s in slots:
                    sid, task = s.get("id"), s.get("id_task", -1)
                    dec = (s.get("next_token") or [{}])[0].get("n_decoded", 0)
                    p_task, p_dec = prev.get(sid, (task, dec))
                    delta += max(0, dec - p_dec) if task == p_task else dec
                    prev[sid] = (task, dec)
                history.append((now, delta))
                history[:] = [(t, d) for (t, d) in history if now - t <= 4]
                span = (now - history[0][0]) if len(history) > 1 else 0
                tps = (sum(d for _, d in history) / span) if span > 0.2 else 0
                running = sum(bool(x.get("is_processing")) for x in slots)
                surya_line = f"~{tps:,.0f} tok/s   ({len(slots)} slots, {running} running)"
            else:
                surya_line = "(server busy)"

        # --- Local LM tokens/sec (from the app's per-call .llm_usage.json) ---
        stats = read_json(STATS_PATH)
        ocr_stats = read_json(OCR_PATH)
        if stats:
            age = time.time() - stats.get("ts", 0)
            fresh = "" if age < 60 else f"{DIM}"
            llm_line = (f"{fresh}~{stats.get('tps', 0):,.0f} tok/s   {stats.get('model', '?')}   "
                        f"({age:.0f}s ago){RST if fresh else ''}")
        else:
            llm_line = f"{DIM}no extraction recorded yet{RST}"

        if (not once) or iters == WARMUP - 1:
            sys.stdout.write("\033[2J\033[H" if tty else ("-" * 60 + "\n"))
            print(f"App usage — {time.strftime('%H:%M:%S')}" + ("   (Ctrl-C to stop)" if tty else ""))
            if not rows:
                print("App not running — start it with ./run.sh")
                print(f"{DIM}(llama-server spawns on the first OCR.){RST}")
            else:
                print(f"Memory {rss_gb:.1f} GB ({100*rss_gb*GB/memsize:.0f}% of {memsize/GB:.0f} GB)"
                      f"  ·  CPU {cpu:.0f}%  ·  swap {swap_used:.1f} GB used")
                print(f"Surya OCR: {surya_line}")
                if ocr_stats:
                    oage = time.time() - ocr_stats.get("ts", 0)
                    files = ocr_stats.get("files_total", 0)
                    print(f"           {DIM}{files} file{'s' if files != 1 else ''}, "
                          f"{ocr_stats.get('pages_total', 0)} pages OCR'd this session "
                          f"(last file {ocr_stats.get('pages_last', 0)} pages, {oage:.0f}s ago){RST}")
                print(f"Local LM:  {llm_line}")
                for rss_kb, c, cmd, pid in rows:
                    print(f"  {rss_kb/1048576:5.1f} GB  {c:4.0f}%  {label(cmd):<28} pid {pid}")
                free_gb = free_ram_bytes(memsize, pagesize) / GB
                if free_gb < alarm_gb:
                    print(f"{RED}⚠ headroom low: {free_gb:.1f} GB system RAM free — the app is "
                          f"near forcing swap. Lower SURYA_INFERENCE_PARALLEL or close apps.{RST}")

        iters += 1
        if once and iters >= WARMUP:
            break
        time.sleep(interval)
except KeyboardInterrupt:
    print("\nstopped.")
PY
