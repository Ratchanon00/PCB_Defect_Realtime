import sys
import os
import socket
import webbrowser
import threading
import time
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int = 8000) -> int:
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def wait_and_open_browser(url: str, port: int):
    for _ in range(60):
        time.sleep(0.5)
        if is_port_in_use(port):
            try:
                webbrowser.open(url)
            except Exception:
                pass
            break

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print(" [*] PCB DEFECT VISION AI - DASHBOARD SERVER")
    print("=" * 65)
    print(f" URL: {url}")
    print(" Initializing PyTorch CUDA & YOLO11m Engine...")
    print("=" * 65)

    threading.Thread(target=wait_and_open_browser, args=(url, port), daemon=True).start()

    uvicorn.run("backend.server:app", host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
