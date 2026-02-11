from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
import socket
import time


def _is_port_available(host: str, port: int) -> bool:
    """Return True if (host, port) can be bound by this process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _choose_port(host: str, preferred: int, *, max_tries: int = 20) -> int:
    for port in range(preferred, preferred + max_tries):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No free port found in range [{preferred}, {preferred + max_tries - 1}] for host {host}."
    )


def _run_foreground(cmd: list[str], *, cwd: str) -> int:
    """Run cmd in the foreground and ensure child process is cleaned up on Ctrl+C."""
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()

        try:
            proc.wait(timeout=5)
        except Exception:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=5)

        time.sleep(0.2)
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the interactive PARSEC Bokeh app (helper wrapper)."
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("BOKEH_PORT", "5006")))
    parser.add_argument(
        "--address",
        default=os.getenv("BOKEH_ADDRESS", "0.0.0.0"),
        help="Bind address for the Bokeh server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        help=(
            "Extra websocket origin(s) to allow, e.g. 'localhost:5008'. "
            "Repeat this flag to add multiple origins."
        ),
    )
    parser.add_argument(
        "--allow-all-websocket-origins",
        action="store_true",
        help=(
            "Allow websocket connections from any Origin (equivalent to '--allow-websocket-origin=*'). "
            "Convenient for VS Code port-forwarding; less secure if exposed publicly."
        ),
    )
    args = parser.parse_args()

    port = int(args.port)
    address = str(args.address)
    if not _is_port_available(address, port):
        new_port = _choose_port(address, port + 1)
        print(f"Port {port} is in use on {address}; using {new_port} instead.")
        port = new_port

    app_path = Path(__file__).resolve().parent / "bokeh_parsec_app.py"

    extra_env = os.getenv("BOKEH_ALLOW_WS_ORIGIN", "").strip()
    extra_env_origins = [o.strip() for o in extra_env.split(",") if o.strip()]

    allow_origins = [
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        f"0.0.0.0:{port}",
        *extra_env_origins,
        *args.allow_websocket_origin,
    ]
    if args.allow_all_websocket_origins:
        allow_origins.append("*")
    allow_origins = list(dict.fromkeys(allow_origins))

    cmd = [
        sys.executable,
        "-m",
        "bokeh",
        "serve",
        str(app_path),
        "--port",
        str(port),
        "--address",
        str(address),
    ]

    for origin in allow_origins:
        cmd.append(f"--allow-websocket-origin={origin}")

    print("Running:")
    print(" ".join(cmd))
    print(f"\nOpen in browser (via forwarded port): http://127.0.0.1:{port}/bokeh_parsec_app")
    print(
        "\nTip: If you see a blank page and the server logs say "
        "\"Refusing websocket connection from Origin 'http://localhost:XXXX'\", rerun with "
        "--allow-websocket-origin=localhost:XXXX (or set BOKEH_ALLOW_WS_ORIGIN=localhost:XXXX)."
    )

    return _run_foreground(cmd, cwd=str(app_path.parent))


if __name__ == "__main__":
    raise SystemExit(main())
