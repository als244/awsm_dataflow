"""
Training Dashboard Logger (Persistent)
=======================================
Lightweight client that sends training stats to the dashboard server.
Each logger instance represents one training run, stored persistently.

Usage:
    from dashboard_logger import DashboardLogger

    logger = DashboardLogger(
        url="http://localhost:8501",
        run_id="my_experiment_v3",
        run_name="GPT-124M AdamW lr=6e-4",
        model="nanogpt_124M",
        config={
            "training_config": training_config,
            "model_hyperparams": model_hyperparams,
            "opt_hyperparams": opt_hyperparams,
            "init_model_path": INIT_MODEL_PATH,
            # ... any other initial conditions you want to record
        }
    )

    # Inside training loop:
    logger.log(cur_step_stats)

    # At end:
    logger.close()
"""

import json
import threading
import queue
import urllib.request
import time
import torch


def _sanitize_config(obj):
    """
    Recursively convert a config dict to JSON-safe types.
    Handles torch tensors, numpy arrays, non-serializable objects, etc.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_config(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)):
        return obj
    # torch tensor
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    # torch device / dtype
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    # numpy
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    # Fallback
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


class DashboardLogger:
    """
    Non-blocking logger that sends training stats to the dashboard server.
    Uses a background thread so it never slows down your training loop.
    """

    def __init__(
        self,
        url="http://localhost:8501",
        run_id=None,
        run_name=None,
        model=None,
        config=None,
        batch_interval=1.0,
        enabled=True,
    ):
        """
        Args:
            url: Dashboard server URL
            run_id: Unique identifier for this run. Auto-generated if None.
            run_name: Human-readable display name
            model: Model identifier string
            config: Dict of initial conditions to store with the run.
                    Typically includes training_config, model_hyperparams,
                    opt_hyperparams, init_model_path, etc.
                    Torch tensors and non-serializable values are auto-converted.
            batch_interval: Seconds between batch sends
            enabled: Set False to disable logging entirely (zero overhead)
        """
        self.url = url.rstrip("/")
        self.enabled = enabled
        self.batch_interval = batch_interval
        self.run_id = run_id or f"run_{int(time.time())}_{id(self) % 10000:04d}"
        self.run_name = run_name or self.run_id
        self.model = model
        self._queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._thread = None
        self._registered = False

        if self.enabled:
            self._register_run(config)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _register_run(self, config=None):
        """Register this run with the server, including sanitized config."""
        safe_config = _sanitize_config(config) if config else None
        try:
            data = json.dumps({
                "run_id": self.run_id,
                "name": self.run_name,
                "model": self.model,
                "config": safe_config,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.url}/api/runs",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            self._registered = True
        except Exception:
            pass  # Server not up yet, will auto-create on first log

    def log(self, step_stats: dict):
        """
        Log a step's stats. Non-blocking — drops silently if queue is full.
        Automatically injects run_id and sanitizes values.
        """
        if not self.enabled:
            return
        clean = {"run_id": self.run_id, "run_name": self.run_name}
        if self.model:
            clean["model"] = self.model
        for k, v in step_stats.items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    clean[k] = str(v)
        try:
            self._queue.put_nowait(clean)
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop_event.is_set():
            time.sleep(self.batch_interval)
            self._flush()
        self._flush()

    def _flush(self):
        batch = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch):
        try:
            data = json.dumps(batch).encode("utf-8")
            req = urllib.request.Request(
                f"{self.url}/api/log_batch",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    def close(self):
        """Mark run as completed, flush remaining stats, and stop."""
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        # Mark run as completed
        try:
            req = urllib.request.Request(
                f"{self.url}/api/runs/{self.run_id}/finish",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass