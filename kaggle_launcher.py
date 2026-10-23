"""Open the Streamlit app from a Kaggle notebook.

Kaggle gives you no public URL, so we start Streamlit in the background and open a tunnel.
Needs: Notebook Settings -> Internet ON.   Usage in a notebook cell:

    from kaggle_launcher import launch
    launch()                      # free Cloudflare quick-tunnel, no account needed
    launch(method="ngrok")        # needs a Kaggle secret NGROK_AUTHTOKEN
"""
import os, re, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _start_streamlit(port):
    log = open("/kaggle/working/streamlit.log" if Path("/kaggle/working").exists() else "streamlit.log", "w")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
                             "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
                            stdout=log, stderr=log, cwd=str(ROOT))


def _cloudflared(port):
    exe = Path("/tmp/cloudflared")
    if not exe.exists():
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
        exe.chmod(0o755)
    p = subprocess.Popen([str(exe), "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(120):
        line = p.stdout.readline()
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line or "")
        if m:
            return m.group(0), p
        time.sleep(0.5)
    raise RuntimeError("Could not get a tunnel URL - is Internet enabled for this notebook?")


def _ngrok(port):
    from pyngrok import ngrok
    from smartaudit.config import get_secret
    ngrok.set_auth_token(get_secret("NGROK_AUTHTOKEN"))
    return ngrok.connect(port).public_url, None


def launch(port=8501, method="cloudflare"):
    sys.path.insert(0, str(ROOT))
    from smartaudit.config import load_secrets_into_env
    print("Secrets found:", load_secrets_into_env())
    srv = _start_streamlit(port); time.sleep(6)
    url, tun = (_cloudflared if method == "cloudflare" else _ngrok)(port)
    print("\n✅ Open the app here:", url, "\n(keep this cell running; stop it to shut the app down)")
    try:
        srv.wait()
    except KeyboardInterrupt:
        srv.terminate()
        if tun: tun.terminate()
