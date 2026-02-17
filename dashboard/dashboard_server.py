"""
Training Dashboard Server (Persistent)
=======================================
Stores all training runs in a SQLite database. Supports browsing
past runs, live updates via WebSocket, and run metadata.

Usage:
    python dashboard_server.py [--port 8501] [--host 0.0.0.0] [--db dashboard.db]

Then open http://localhost:8501 in your browser.
"""

import argparse
import json
import sqlite3
import threading
import time
import struct
import hashlib
import base64
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = "dashboard.db"
db_lock = threading.Lock()


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            name TEXT,
            model TEXT,
            status TEXT DEFAULT 'running',
            config TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            total_steps INTEGER DEFAULT 0,
            final_loss REAL,
            total_tokens INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step_num INTEGER NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            UNIQUE(run_id, step_num)
        );
        CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step_num);
    """)
    conn.commit()
    conn.close()


def db_create_run(run_id, name=None, model=None, config=None):
    with db_lock:
        conn = get_db()
        conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, name, model, config) VALUES (?, ?, ?, ?)",
            (run_id, name or run_id, model, json.dumps(config) if config else None)
        )
        conn.commit()
        conn.close()


def db_log_step(run_id, step_num, data):
    with db_lock:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO steps (run_id, step_num, data) VALUES (?, ?, ?)",
            (run_id, step_num, json.dumps(data))
        )
        loss = data.get("avg_loss")
        total_tokens = data.get("total_tokens", 0)
        conn.execute(
            """UPDATE runs SET
                updated_at = datetime('now'),
                total_steps = MAX(total_steps, ?),
                final_loss = ?,
                total_tokens = MAX(total_tokens, ?),
                status = 'running'
            WHERE run_id = ?""",
            (step_num, loss, total_tokens, run_id)
        )
        conn.commit()
        conn.close()


def db_finish_run(run_id):
    with db_lock:
        conn = get_db()
        conn.execute(
            "UPDATE runs SET status = 'completed', updated_at = datetime('now') WHERE run_id = ?",
            (run_id,)
        )
        conn.commit()
        conn.close()


def db_delete_run(run_id):
    with db_lock:
        conn = get_db()
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()


def db_rename_run(run_id, new_name):
    with db_lock:
        conn = get_db()
        conn.execute("UPDATE runs SET name = ? WHERE run_id = ?", (new_name, run_id))
        conn.commit()
        conn.close()


def db_get_runs():
    conn = get_db()
    rows = conn.execute("SELECT * FROM runs ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_run(run_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_get_steps(run_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT step_num, data FROM steps WHERE run_id = ? ORDER BY step_num", (run_id,)
    ).fetchall()
    conn.close()
    return [json.loads(r["data"]) for r in rows]


# ─── WebSocket ────────────────────────────────────────────────────────────────
connected_clients = []
ws_lock = threading.Lock()


def ws_encode_frame(payload: bytes, opcode=0x1) -> bytes:
    frame = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(payload)
    return bytes(frame)


def ws_decode_frame(data: bytes):
    if len(data) < 2:
        return None, None, 0
    opcode = data[0] & 0x0F
    masked = bool(data[1] & 0x80)
    length = data[1] & 0x7F
    offset = 2
    if length == 126:
        if len(data) < 4: return None, None, 0
        length = struct.unpack(">H", data[2:4])[0]
        offset = 4
    elif length == 127:
        if len(data) < 10: return None, None, 0
        length = struct.unpack(">Q", data[2:10])[0]
        offset = 10
    if masked:
        if len(data) < offset + 4 + length: return None, None, 0
        mask = data[offset:offset+4]; offset += 4
        payload = bytearray(data[offset:offset+length])
        for i in range(length): payload[i] ^= mask[i % 4]
        payload = bytes(payload)
    else:
        if len(data) < offset + length: return None, None, 0
        payload = data[offset:offset+length]
    return opcode, payload, offset + length


def ws_handshake(request_text: str) -> str:
    key = None
    for line in request_text.split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
    if not key: return None
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-5AB0A7D6AF4E").encode()).digest()
    ).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )


def handle_ws_client(conn, addr):
    entry = {"conn": conn, "run_id": None}
    with ws_lock:
        connected_clients.append(entry)
    try:
        buf = b""
        while True:
            try:
                data = conn.recv(4096)
                if not data: break
                buf += data
                while buf:
                    opcode, payload, consumed = ws_decode_frame(buf)
                    if opcode is None: break
                    buf = buf[consumed:]
                    if opcode == 0x8: return
                    elif opcode == 0x9:
                        conn.sendall(ws_encode_frame(payload, opcode=0xA))
                    elif opcode == 0x1:
                        try:
                            msg = json.loads(payload.decode("utf-8"))
                            if msg.get("type") == "subscribe":
                                entry["run_id"] = msg.get("run_id")
                        except: pass
            except: break
    finally:
        with ws_lock:
            if entry in connected_clients:
                connected_clients.remove(entry)
        try: conn.close()
        except: pass


def broadcast_to_run(run_id, msg_str):
    frame = ws_encode_frame(msg_str.encode("utf-8"))
    with ws_lock:
        dead = []
        for e in connected_clients:
            if e["run_id"] == run_id or e["run_id"] is None:
                try: e["conn"].sendall(frame)
                except: dead.append(e)
        for d in dead: connected_clients.remove(d)


def broadcast_all(msg_str):
    frame = ws_encode_frame(msg_str.encode("utf-8"))
    with ws_lock:
        dead = []
        for e in connected_clients:
            try: e["conn"].sendall(frame)
            except: dead.append(e)
        for d in dead: connected_clients.remove(d)


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = None


class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Log POST requests for debugging
        pass

    def _log_request(self, method, path, detail=""):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {method} {path} {detail}")

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_OPTIONS(self):
        self.send_response(200)
        for h, v in [("Access-Control-Allow-Origin","*"),
                      ("Access-Control-Allow-Methods","GET,POST,PUT,DELETE,OPTIONS"),
                      ("Access-Control-Allow-Headers","Content-Type")]:
            self.send_header(h, v)
        self.end_headers()

    def _run_id_from_path(self, path):
        """Extract and URL-decode the run_id from /api/runs/{run_id}[/...]"""
        parts = path.split("/")
        # /api/runs/{run_id} -> parts = ['', 'api', 'runs', '{run_id}']
        # /api/runs/{run_id}/steps -> parts = ['', 'api', 'runs', '{run_id}', 'steps']
        if len(parts) >= 4:
            return unquote(parts[3])
        return None

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/dashboard"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif p == "/ws":
            req = f"GET {self.path} HTTP/1.1\r\n"
            for k, v in self.headers.items(): req += f"{k}: {v}\r\n"
            resp = ws_handshake(req)
            if resp:
                self.wfile.write(resp.encode("utf-8")); self.wfile.flush()
                handle_ws_client(self.connection, self.client_address)
            else: self.send_error(400)
        elif p == "/api/runs":
            self._json(db_get_runs())
        elif p.startswith("/api/runs/") and p.endswith("/steps"):
            rid = self._run_id_from_path(p)
            self._log_request("GET", f"/api/runs/.../steps", f"run={rid}")
            self._json(db_get_steps(rid))
        elif p.startswith("/api/runs/"):
            rid = self._run_id_from_path(p)
            run = db_get_run(rid)
            self._json(run if run else {"error":"not found"}, 200 if run else 404)
        else:
            self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/runs":
            b = json.loads(self._body())
            rid = b.get("run_id", f"run_{int(time.time())}")
            self._log_request("POST", "/api/runs", f"run={rid} name={b.get('name')} has_config={b.get('config') is not None}")
            db_create_run(rid, b.get("name"), b.get("model"), b.get("config"))
            broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"run_id": rid})
        elif p == "/api/log":
            b = json.loads(self._body())
            rid = b.get("run_id", "default")
            sn = b.get("step_num", 0)
            self._log_request("POST", "/api/log", f"run={rid} step={sn} keys={list(b.keys())[:8]}")
            db_create_run(rid, name=b.get("run_name", rid), model=b.get("model"))
            db_log_step(rid, sn, b)
            broadcast_to_run(rid, json.dumps({"type":"step","run_id":rid,"data":b}))
            broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"status":"ok"})
        elif p == "/api/log_batch":
            items = json.loads(self._body())
            if not items: self._json({"status":"ok"}); return
            rid = items[0].get("run_id", "default")
            self._log_request("POST", "/api/log_batch", f"run={rid} count={len(items)} step_nums={[it.get('step_num','?') for it in items[:5]]}")
            db_create_run(rid, name=items[0].get("run_name",rid), model=items[0].get("model"))
            for it in items:
                db_log_step(rid, it.get("step_num",0), it)
                broadcast_to_run(rid, json.dumps({"type":"step","run_id":rid,"data":it}))
            broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"status":"ok"})
        elif p.startswith("/api/runs/") and p.endswith("/finish"):
            rid = self._run_id_from_path(p)
            db_finish_run(rid)
            broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"status":"ok"})
        else: self.send_error(404)

    def do_PUT(self):
        p = urlparse(self.path).path
        if p.startswith("/api/runs/"):
            rid = self._run_id_from_path(p)
            b = json.loads(self._body())
            if "name" in b:
                db_rename_run(rid, b["name"])
                broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"status":"ok"})
        else: self.send_error(404)

    def do_DELETE(self):
        p = urlparse(self.path).path
        if p.startswith("/api/runs/"):
            rid = self._run_id_from_path(p)
            db_delete_run(rid)
            broadcast_all(json.dumps({"type":"runs_updated","data":db_get_runs()}))
            self._json({"status":"ok"})
        else: self.send_error(404)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global DASHBOARD_HTML, DB_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--db", type=str, default="dashboard.db")
    args = parser.parse_args()
    DB_PATH = args.db

    html_path = Path(__file__).parent / "dashboard.html"
    if not html_path.exists():
        print(f"ERROR: {html_path} not found!"); return
    DASHBOARD_HTML = html_path.read_text()

    init_db()
    server = ThreadedHTTPServer((args.host, args.port), DashboardHandler)
    print(f"╔═══════════════════════════════════════════════════════╗")
    print(f"║  Training Dashboard                                  ║")
    print(f"║  http://localhost:{args.port:<38}║")
    print(f"║  Database: {args.db:<44}║")
    print(f"╚═══════════════════════════════════════════════════════╝")
    try: server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down..."); server.shutdown()

if __name__ == "__main__":
    main()