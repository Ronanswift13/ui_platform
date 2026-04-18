"""Minimal preview server for cockpit dashboard (runs within ui/ directory)."""
import http.server
import os
import sys

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

UI_DIR = Path(__file__).parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


class PreviewHandler(http.server.SimpleHTTPHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/cockpit":
            try:
                t = env.get_template("pages/cockpit.html")
                html = t.render(
                    request=None,
                    active_tab="cockpit",
                    version="4.0.0",
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            except Exception as e:
                self.send_error(500, str(e))
        elif self.path.startswith("/static/"):
            self.path = self.path[1:]  # strip leading /
            file_path = STATIC_DIR.parent / self.path
            if file_path.exists():
                self.send_response(200)
                ct = "text/css" if self.path.endswith(".css") else \
                     "application/javascript" if self.path.endswith(".js") else \
                     "application/octet-stream"
                self.send_header("Content-Type", ct)
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
            else:
                self.send_error(404)
        elif self.path == "/api/plugins/list":
            import json
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps([]).encode())
        else:
            self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    os.chdir(str(UI_DIR))
    server = http.server.HTTPServer(("127.0.0.1", port), PreviewHandler)
    print(f"Preview server on http://127.0.0.1:{port}")
    server.serve_forever()
