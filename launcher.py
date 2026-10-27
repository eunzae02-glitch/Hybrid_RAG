"""Single entry point for the bundled local web application."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser


BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".local"
STATE_FILE = STATE_DIR / "server.json"
STOP_FILE = STATE_DIR / "stop.json"
LOG_DIR = BASE_DIR / "logs"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
STARTUP_TIMEOUT = 90


class InstanceLock:
    def __init__(self):
        self.file = None

    def acquire(self) -> bool:
        import msvcrt

        self.file = (STATE_DIR / "server.lock").open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            self.file.close()
            self.file = None
            return False

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(state: dict):
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE_FILE)


def is_healthy(url: str) -> bool:
    try:
        with HTTP.open(f"{url}/_stcore/health", timeout=0.7) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def find_free_port() -> int:
    for port in range(8501, 8521):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            if sys.platform == "win32":
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                candidate.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("8501~8520 포트를 사용할 수 없습니다. 다른 로컬 서버를 확인해 주세요.")


def notify_error(message: str):
    logging.error(message)
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(
            None,
            f"{message}\n\n실행 기록: {LOG_DIR / 'server.log'}",
            "논문RAG 실행 오류",
            0x10,
        )
    else:
        print(message, file=sys.stderr)


def reuse_running_server(open_browser: bool) -> int:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        state = read_json(STATE_FILE)
        url = state.get("url", "")
        if state.get("status") == "ready" and url and is_healthy(url):
            if open_browser:
                webbrowser.open(url)
            logging.info("Reused running server: %s", url)
            return 0
        time.sleep(0.3)
    raise RuntimeError("실행 중인 서버가 응답하지 않습니다. 잠시 후 다시 실행해 주세요.")


def stop_process(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_server(open_browser: bool) -> int:
    python = BASE_DIR / "runtime" / "python.exe"
    app = BASE_DIR / "app.py"
    if not python.is_file() or not app.is_file():
        raise RuntimeError("runtime 폴더 또는 app.py를 찾을 수 없습니다. 프로젝트 폴더 전체를 함께 보관해 주세요.")

    port = find_free_port()
    token = uuid.uuid4().hex
    state = {"url": f"http://127.0.0.1:{port}", "token": token, "status": "starting", "owner_pid": os.getpid()}
    STOP_FILE.unlink(missing_ok=True)
    write_state(state)

    log_path = LOG_DIR / "server.log"
    if log_path.exists() and log_path.stat().st_size > 5_000_000:
        log_path.replace(LOG_DIR / "server.previous.log")

    command = [
        str(python), "-X", "utf8", "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)

    process = None
    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command, cwd=BASE_DIR, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            state["pid"] = process.pid
            write_state(state)
            started = time.monotonic()
            while not is_healthy(state["url"]):
                if process.poll() is not None:
                    raise RuntimeError("로컬 서버를 시작하지 못했습니다. 실행 기록을 확인해 주세요.")
                if time.monotonic() - started >= STARTUP_TIMEOUT:
                    raise RuntimeError("서버 시작 대기 시간을 초과했습니다.")
                time.sleep(0.2)

            state["status"] = "ready"
            write_state(state)
            logging.info("Server ready in %.2fs: %s (PID %s)", time.monotonic() - started, state["url"], process.pid)
            if open_browser:
                webbrowser.open(state["url"])

            while process.poll() is None:
                if read_json(STOP_FILE).get("token") == token:
                    logging.info("Shutdown requested from the application")
                    stop_process(process)
                    return 0
                time.sleep(0.5)

            raise RuntimeError("로컬 서버가 종료되었습니다. 실행 파일을 다시 열어 주세요.")
    finally:
        if process is not None:
            stop_process(process)
        if read_json(STATE_FILE).get("token") == token:
            STATE_FILE.unlink(missing_ok=True)
            STOP_FILE.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="논문RAG 로컬 웹 실행")
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    STATE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOG_DIR / "launcher.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    lock = InstanceLock()
    try:
        if not lock.acquire():
            return reuse_running_server(not options.no_browser)
        return run_server(not options.no_browser)
    except Exception as error:
        logging.exception("Launcher failed")
        notify_error(str(error))
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
