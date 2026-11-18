"""Lightweight web server to host the PathSnap Evaluator Frontend (Laptop 2).

Usage:
    python frontend/serve.py --port 3000

Then open in browser:
    http://localhost:3000
"""

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

PORT = 3000
if len(sys.argv) > 2 and sys.argv[1] == "--port":
    PORT = int(sys.argv[2])

FRONTEND_DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

def main():
    print(f"==================================================")
    print(f"PathSnap Evaluator Frontend (Laptop 2 Client)")
    print(f"Serving at: http://localhost:{PORT}")
    print(f"Serving files from: {FRONTEND_DIR}")
    print(f"Press Ctrl+C to stop.")
    print(f"==================================================")
    
    # Try to open in browser automatically
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down frontend server.")

if __name__ == "__main__":
    main()
