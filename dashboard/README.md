# Training Dashboard

A real-time training dashboard with persistent run history. All runs are stored in a SQLite database — browse, compare, rename, and delete past runs from a collapsible sidebar.

**Zero external dependencies.** Pure Python stdlib + SQLite. Nothing to `pip install`.

---

## Directory Layout

Place the `dashboard/` folder alongside your training script:

```
your_project/
├── train.py
├── model_dims.json
├── dashboard/
│   ├── dashboard_server.py      # Server: HTTP + WebSocket + SQLite
│   ├── dashboard_logger.py      # Client: import into train.py
│   ├── dashboard.html           # Frontend: served by the server
│   └── README.md                # This file
├── fineweb10B/
│   └── ...
└── ...
```

## Requirements

- **Python 3.8+** (uses only stdlib: `http.server`, `sqlite3`, `threading`, `json`, `urllib`)
- **No pip packages** — nothing to install
- The logger imports `torch` only for serializing tensors in your config dict; if you don't pass tensors, it has no hard dependency on torch

---

## Quick Start

### 1. Start the dashboard server (separate terminal)

```bash
cd your_project/
python dashboard/dashboard_server.py --port 8501
```

You should see:

```
╔═══════════════════════════════════════════════════════╗
║  Training Dashboard                                  ║
║  http://localhost:8501                                ║
║  Database: dashboard.db                              ║
╚═══════════════════════════════════════════════════════╝
```

Open **http://localhost:8501** in your browser.

### 2. Add to your training script

Add these lines to `train.py` (shown relative to your existing code):

```python
# ─── At the top, with other imports ───────────────────
from dashboard.dashboard_logger import DashboardLogger

# ─── After opt_hyperparams is defined ─────────────────
dashboard = DashboardLogger(
    url="http://localhost:8501",
    run_id=RUN_NAME,                              # unique ID for this run
    run_name=f"{MODEL_CHOICE} {RUN_NAME}",        # display name in sidebar
    model=MODEL_CHOICE,                           # model tag
    config={                                       # initial conditions (viewable in Run Config tab)
        "model_dims": model_dims,
        "training_config": training_config,
        "model_hyperparams": model_hyperparams,
        "opt_hyperparams": opt_hyperparams,
        "init_model_path": INIT_MODEL_PATH,
    },
)

# ─── Inside training loop, after step_stats[step_num] = cur_step_stats ──
    cur_step_stats["lr"] = opt_hyperparams["lr"]  # include LR in logged data
    dashboard.log(cur_step_stats)

# ─── After the training loop ends, before cleanup ────
dashboard.close()
```

### 3. Run training as normal

```bash
python train.py
```

The dashboard will populate in real-time. That's it.

---

## How It Works

### Logging behavior

- The logger runs a **background thread** that batches and sends stats via HTTP POST every 1 second (configurable via `batch_interval`).
- If the server is unreachable (not started, crashed, wrong port), the logger **silently drops data** — it will never raise an exception, block, or slow down your training loop.
- When you call `dashboard.close()`, it flushes any remaining buffered steps, then sends a "finish" signal that marks the run as `completed` in the sidebar.
- If your training crashes before `close()` is called, the run stays marked as `running`. All steps logged up to that point are still saved in the database.

### Run identity

- Each `run_id` is a unique key in the database. If you reuse the same `run_id`, new steps **append** to the existing run (useful for resuming training).
- If you omit `run_id`, one is auto-generated from the timestamp.
- `run_name` is purely cosmetic — it's the label shown in the sidebar and can be changed at any time via the rename button in the UI.

### Server behavior

- The server is a threaded Python HTTP server that also handles WebSocket upgrades for live chart updates.
- It serves the dashboard HTML at `/` and exposes a REST API under `/api/`.
- The server can be started **before or after** training — the logger will auto-create the run on first log if the server wasn't up during `__init__`.
- You can stop and restart the server at any time. All data is in SQLite and persists. The browser auto-reconnects when the server comes back.
- Multiple training runs can log to the same server concurrently (each with a different `run_id`).

### Browser behavior

- Open `http://localhost:8501` in any browser.
- The **sidebar** lists all runs sorted by most recently updated. Click a run to load its charts.
- The **Metrics** tab shows 7 live charts. The **Run Config** tab shows the initial conditions you passed in the `config` dict.
- The **X-Axis selector** (Step / Time / FLOPS / Tokens / Sequences) re-renders all charts instantly — useful for comparing efficiency across different batch sizes or hardware.
- Hover over any run in the sidebar to see **rename** (✎) and **delete** (✕) buttons.
- If a run is currently `running`, its charts update live as new steps arrive via WebSocket.
- If you open the dashboard mid-training, it loads the full history from the database immediately.

---

## Database

### Location

The database file (`dashboard.db`) is created in the **working directory where you launch the server**. Control it explicitly with:

```bash
python dashboard/dashboard_server.py --db dashboard/dashboard.db
```

### Schema

**`runs`** — one row per training run:

| Column | Type | Description |
|---|---|---|
| `run_id` | TEXT PRIMARY KEY | Your `RUN_NAME` value (or auto-generated) |
| `name` | TEXT | Display name shown in sidebar |
| `model` | TEXT | Model identifier (e.g. `nanogpt_124M`) |
| `status` | TEXT | `running` or `completed` |
| `config` | TEXT | JSON blob — your full config dict (model_dims, training_config, etc.) |
| `created_at` | TEXT | ISO datetime when the run was first created |
| `updated_at` | TEXT | ISO datetime, updates on every step |
| `total_steps` | INTEGER | High-water mark step number |
| `final_loss` | REAL | Most recent `avg_loss` value |
| `total_tokens` | INTEGER | High-water mark total tokens processed |

**`steps`** — one row per training step:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Auto-increment |
| `run_id` | TEXT | Foreign key → `runs.run_id` |
| `step_num` | INTEGER | Step number (unique per run) |
| `data` | TEXT | JSON blob of the full `cur_step_stats` dict |
| `created_at` | TEXT | ISO datetime |

There is a `UNIQUE(run_id, step_num)` constraint (upserts on conflict) and an index on `(run_id, step_num)` for fast lookups. The database uses **WAL journal mode** for safe concurrent reads and writes.

### Querying the database directly

You can use the database for offline analysis without the server:

```python
import sqlite3, json

conn = sqlite3.connect("dashboard.db")
conn.row_factory = sqlite3.Row

# List all runs
for r in conn.execute("SELECT run_id, name, status, total_steps, final_loss FROM runs"):
    print(dict(r))

# Load all steps for a specific run
steps = [
    json.loads(row["data"])
    for row in conn.execute(
        "SELECT data FROM steps WHERE run_id = ? ORDER BY step_num",
        ("512k_adamw",)
    )
]

# Extract a specific metric as a list
losses = [s["avg_loss"] for s in steps]

# Load the config for a run
row = conn.execute("SELECT config FROM runs WHERE run_id = ?", ("512k_adamw",)).fetchone()
config = json.loads(row["config"]) if row["config"] else None
```

### Database size

Each step stores the full `cur_step_stats` dict as JSON (~500–800 bytes depending on your fields). Rough sizing:

| Steps | Approx DB size |
|---|---|
| 1,000 | ~1 MB |
| 10,000 | ~8 MB |
| 100,000 | ~80 MB |

If you need to trim old runs, delete them from the UI or directly:

```python
conn.execute("DELETE FROM steps WHERE run_id = ?", ("old_run",))
conn.execute("DELETE FROM runs WHERE run_id = ?", ("old_run",))
conn.commit()
```

---

## Tracked Metrics

These fields from your `cur_step_stats` dict are auto-detected and plotted:

| Field | Chart | Notes |
|---|---|---|
| `avg_loss` | Training Loss | Raw loss (solid line) |
| `loss_smoothed` | Training Loss | EMA smoothed (dashed overlay) |
| `step_tokens_per_sec` | Tokens / sec | Per-step throughput |
| `step_throughput_tflops` | Effective TFLOPS | Per-step compute throughput |
| `lr` | Learning Rate | Must be added manually: `cur_step_stats["lr"] = opt_hyperparams["lr"]` |
| `step_tokens` | Tokens per Step | Token count per step |
| `step_duration` | Step Duration | Wall-clock seconds per step |
| `total_tokens` | Cumulative Tokens | Running total (plotted in millions) |

### X-Axis options

All charts share a global x-axis selector. The following fields are used as x-axis values:

| Button | Field | Format examples |
|---|---|---|
| **Step** | `step_num` | 1, 100, 1000 |
| **Time** | `total_train_time` | 45s, 12.3m, 1.50h |
| **FLOPS** | `total_flops_cost` | 5.2T, 1.3P, 0.8E |
| **Tokens** | `total_tokens` | 524K, 100.7M, 1.20B |
| **Sequences** | `total_seqs` | 1.0K, 50.2K, 1.1M |

All of these fields already exist in your `cur_step_stats` dict — no changes needed.

---

## Server CLI

```bash
python dashboard/dashboard_server.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--port` | `8501` | HTTP port |
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` = accessible from LAN) |
| `--db` | `dashboard.db` | SQLite database file path |

---

## Logger Configuration

```python
dashboard = DashboardLogger(
    url="http://localhost:8501",   # Server URL
    run_id="my_run",              # Unique ID (auto-generated if omitted)
    run_name="My Experiment",     # Sidebar display name
    model="nanogpt_124M",        # Model tag (optional)
    config={...},                 # Initial conditions dict (optional)
    batch_interval=1.0,           # Seconds between HTTP sends (default 1.0)
    enabled=True,                 # Set False to disable entirely (zero overhead)
)
```

To disable logging without removing code (e.g. for benchmarking):

```python
dashboard = DashboardLogger(enabled=False)
```

All calls to `dashboard.log()` and `dashboard.close()` become no-ops with zero overhead.

---

## REST API

The server exposes a JSON API if you want to build custom tooling:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/runs` | List all runs (sorted by most recently updated) |
| `GET` | `/api/runs/{id}` | Get run metadata including config |
| `GET` | `/api/runs/{id}/steps` | Get all step data for a run |
| `POST` | `/api/runs` | Create a new run |
| `POST` | `/api/log` | Log a single step |
| `POST` | `/api/log_batch` | Log a batch of steps |
| `POST` | `/api/runs/{id}/finish` | Mark run as completed |
| `PUT` | `/api/runs/{id}` | Rename a run (`{"name": "new name"}`) |
| `DELETE` | `/api/runs/{id}` | Delete a run and all its step data |

---

## Files

| File | Purpose |
|---|---|
| `dashboard_server.py` | HTTP server, WebSocket handler, SQLite database, REST API |
| `dashboard_logger.py` | Non-blocking client logger (import into your training script) |
| `dashboard.html` | Single-file frontend with sidebar, charts, config viewer (served by the server) |
| `README.md` | This file |

---

## Remote Access

If you're training on a remote server (cloud VM, cluster node, etc.) and want to view the dashboard from your local browser, there are two options.

### Option 1: SSH Port Forwarding (recommended)

This is the simplest and most secure approach. Everything stays as `localhost` in your code — SSH tunnels the port to your local machine.

On your **local machine**, connect with:

```bash
ssh -L 8501:localhost:8501 user@remote-server
```

Then open `http://localhost:8501` in your local browser. No changes to the training script or server config. The tunnel stays open as long as the SSH session is active.

If you want the tunnel in the background:

```bash
ssh -fNL 8501:localhost:8501 user@remote-server
```

To kill it later: `kill $(lsof -ti:8501)` on your local machine.

### Option 2: Bind to the server's IP directly

If SSH tunneling isn't practical (e.g. you're on a shared cluster with a web-accessible hostname), you can point everything at the server's real IP or hostname.

The server already binds to `0.0.0.0` by default, so no server-side changes are needed — just start it normally:

```bash
python dashboard/dashboard_server.py --port 8501
```

Update the logger URL in `train.py` to use the machine's hostname or IP:

```python
dashboard = DashboardLogger(
    url="http://my-gpu-box:8501",       # hostname
    # or url="http://10.0.1.42:8501",   # IP address
    ...
)
```

Open `http://my-gpu-box:8501` (or the IP) in your local browser.

**Firewall note:** Port 8501 must be reachable from your local machine. If there's a firewall, you'll need to open it (e.g. `ufw allow 8501` on Ubuntu, or configure your cloud provider's security group). The SSH tunnel approach avoids this entirely.

### Option 3: Running on a multi-node cluster

If your training script and dashboard server run on different nodes, start the server on a node with a stable hostname (e.g. a login node or head node), and point the logger at it:

```bash
# On the head node:
python dashboard/dashboard_server.py --port 8501 --db /shared/storage/dashboard.db

# In train.py on the compute node:
dashboard = DashboardLogger(url="http://head-node:8501", ...)
```

Use a shared filesystem path for `--db` if you want the database accessible from multiple nodes.

---

## FAQ

**Q: What happens if I forget to start the server before training?**
The logger silently fails to connect. Once you start the server, subsequent steps will begin logging. Steps from before the server was started are lost (they were dropped by the logger's background thread).

**Q: What if training crashes?**
All steps logged up to that point are saved in the database. The run will show as `running` in the sidebar since `close()` was never called. You can manually mark it completed or just ignore it.

**Q: Can I run multiple trainings logging to the same server?**
Yes. Each must have a different `run_id`. They'll appear as separate entries in the sidebar and their live updates are routed independently via WebSocket subscriptions.

**Q: Can I access the dashboard from another machine?**
Yes. See the [Remote Access](#remote-access) section above. The easiest way is `ssh -L 8501:localhost:8501 user@remote-server`, then open `http://localhost:8501` locally.

**Q: How do I reset everything?**
Delete the `dashboard.db` file (or whichever path you used with `--db`). The server will create a fresh database on next startup.

**Q: Can I add custom metrics?**
Yes. Any key you add to `cur_step_stats` is stored in the database as part of the step JSON blob. It won't auto-plot on the dashboard charts, but it's fully preserved and queryable from the database for custom analysis.

**Q: What if I reuse a `run_id`?**
New steps are appended (upserted by step number). The config is only written on first creation (`INSERT OR IGNORE`), so the original config is preserved. This makes it safe to resume a run.