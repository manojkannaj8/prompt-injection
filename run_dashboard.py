"""Start the prompt-injection defence dashboard.

    python run_dashboard.py            opens http://localhost:8501 in your browser
    python run_dashboard.py --port 8600
    python run_dashboard.py --no-browser

Runs Streamlit from the ui/ folder so that its theme settings are picked up.
The dashboard only reads the research files; it writes nothing outside ui/cache.
"""
import argparse
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

UI = Path(__file__).resolve().parent / "ui"


def main():
    ap = argparse.ArgumentParser(description="Start the prompt-injection defence dashboard.")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = ap.parse_args()
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("Streamlit is not installed. Run:  pip install -r ui/requirements.txt")
    url = f"http://localhost:{args.port}"
    print(f"Dashboard starting at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    # Local use only: listen on localhost. This is set here, not in ui/.streamlit/config.toml, because that
    # file is also read when the app runs on a host, where the server must be reachable from outside.
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port), "--server.headless", "true",
           "--server.address", "localhost"]
    try:
        sys.exit(subprocess.call(cmd, cwd=UI))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
