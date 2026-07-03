#!/usr/bin/env bash
#
# Live Surya (llama-server) OCR usage: memory, parallel slots, and output tokens/sec.
#
#   ./surya_usage.sh          # live, refresh every 1s (Ctrl-C to stop)
#   ./surya_usage.sh 2        # refresh every 2s
#   ./surya_usage.sh once     # a few quick samples, then exit
#
# tokens/sec is the OCR generation speed, measured from the growth of the server's
# per-slot decoded-token counts over time (llama-server's /metrics endpoint is off,
# so this is the reliable way). It finds the dynamic llama-server port itself.

exec python3 - "${1:-1}" <<'PY'
import sys, time, json, re, subprocess, urllib.request

arg = sys.argv[1] if len(sys.argv) > 1 else ""
once = arg == "once"
interval = 0.5 if once else (float(arg) if arg else 1.0)
tty = sys.stdout.isatty()


def find_server():
    pids = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()
    if not pids:
        return None, None
    pid = pids[0]
    lsof = subprocess.run(["lsof", "-nP", "-p", pid], capture_output=True, text=True).stdout
    match = re.search(r"127\.0\.0\.1:(\d+)", lsof)
    return pid, (match.group(1) if match else None)


def rss_gb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()
    return f"{int(out) / 1048576:.1f}" if out.isdigit() else "?"


prev, history, iters = {}, [], 0
try:
    while True:
        pid, port = find_server()
        now = time.time()
        sys.stdout.write("\033[2J\033[H" if tty else ("-" * 60 + "\n"))
        print(f"Surya OCR usage — {time.strftime('%H:%M:%S')}" + ("   (Ctrl-C to stop)" if tty else ""))

        slots = None
        if pid and port:
            try:
                slots = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=2).read())
            except Exception:
                print(f"pid {pid} · port {port} · (server busy)")
        if not pid or not port:
            print("llama-server not running (it spawns on first OCR).")

        if slots is not None:
            delta = 0
            for s in slots:
                sid, task, dec = s["id"], s["id_task"], s["next_token"][0]["n_decoded"]
                p_task, p_dec = prev.get(sid, (task, dec))
                delta += max(0, dec - p_dec) if task == p_task else dec
                prev[sid] = (task, dec)
            history.append((now, delta))
            history[:] = [(t, d) for (t, d) in history if now - t <= 4]
            span = (now - history[0][0]) if len(history) > 1 else 0
            tps = (sum(d for _, d in history) / span) if span > 0.2 else 0
            running = sum(x["is_processing"] for x in slots)
            print(f"pid {pid} · port {port} · RSS {rss_gb(pid)} GB")
            print(f"{len(slots)} slots · {running} running · ~{tps:,.0f} output tok/s")
            for x in slots:
                tag = "RUN " if x["is_processing"] else "idle"
                print(f"  slot {x['id']}: {tag} {x['n_prompt_tokens']:>5} -> {x['next_token'][0]['n_decoded']:>5} tok")

        iters += 1
        if once and iters >= 5:
            break
        time.sleep(interval)
except KeyboardInterrupt:
    print("\nstopped.")
PY
