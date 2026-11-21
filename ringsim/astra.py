"""GPT-6.2-Astra through the locally authenticated Codex app-server (JSON-RPC over stdio).

Tools are disabled: the model can only read the turn input and return JSON matching the schema.
Every turn records wall-clock latency. Threads are ephemeral and never persisted.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import subprocess
import tempfile
import threading
import time

import numpy as np
from PIL import Image

MODEL = "gpt-6.2-astra"
TURN_TIMEOUT = 300.0


def png_part(array) -> dict:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", optimize=False)
    return {"type": "image", "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def depth_png(depth, max_m=2.5):
    return np.uint8(np.clip(depth / max_m, 0, 1) * 255)


class AstraError(RuntimeError):
    pass


class Astra:
    """One app-server process. Threads are cheap; create one per episode or per decision."""

    def __init__(self, effort: str = "low", model: str = MODEL):
        self.effort, self.model = effort, model
        self.tmp = tempfile.TemporaryDirectory(prefix="ringsim-astra-")
        cmd = ["codex", "app-server", "--stdio", "-c", 'web_search="disabled"']
        for feature in ("shell_tool", "unified_exec", "apps", "plugins"):
            cmd += ["--disable", feature]
        self.proc = subprocess.Popen(cmd, cwd=self.tmp.name, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)
        self.inbox: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()
        self.seq = 0
        self.rpc("initialize", {"clientInfo": {"name": "ringsim", "title": "Reaction Ring exam",
                                               "version": "1.0"}, "capabilities": {}})
        self._send({"method": "initialized"})

    # ---- transport --------------------------------------------------------------------------------
    def _pump(self):
        for line in self.proc.stdout:
            try:
                self.inbox.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.inbox.put(None)

    def _send(self, msg):
        with self.lock:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _next(self, deadline):
        try:
            msg = self.inbox.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as e:
            raise AstraError("timed out waiting for the model") from e
        if msg is None:
            raise AstraError("app-server exited")
        if "method" in msg and "id" in msg:          # server->client request: refuse it
            self._send({"id": msg["id"], "error": {"code": -32601, "message": "not supported"}})
            raise AstraError("model attempted a tool/approval request: " + msg["method"])
        return msg

    def rpc(self, method, params, timeout=90.0):
        self.seq += 1
        rid = self.seq
        self._send({"id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._next(deadline)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise AstraError(f"{method}: {msg['error']}")
                return msg["result"]

    # ---- model use ----------------------------------------------------------------------------------
    def new_thread(self, instructions: str, developer: str = "Respond with JSON only.") -> str:
        res = self.rpc("thread/start", {
            "model": self.model, "ephemeral": True, "baseInstructions": instructions,
            "developerInstructions": developer, "cwd": self.tmp.name,
            "approvalPolicy": "never", "sandbox": "read-only",
            "config": {"model_reasoning_effort": self.effort, "mcp_servers": {}}})
        if res.get("model") != self.model:
            raise AstraError(f"served model {res.get('model')!r}, expected {self.model!r}")
        return res["thread"]["id"]

    def ask(self, thread: str, inputs: list, schema: dict, timeout=TURN_TIMEOUT):
        """Run one turn; returns (parsed JSON, info) where info has latency and token usage."""
        t0 = time.monotonic()
        self.rpc("turn/start", {"threadId": thread, "input": inputs, "effort": self.effort,
                                "outputSchema": schema})
        deadline = t0 + timeout
        reply, usage = None, None
        while True:
            msg = self._next(deadline)
            method, params = msg.get("method"), msg.get("params", {})
            if method == "model/rerouted":
                raise AstraError("provider rerouted the request to another model")
            if method in ("item/started", "item/completed"):
                kind = params.get("item", {}).get("type")
                if kind not in ("agentMessage", "userMessage", "reasoning"):
                    raise AstraError("unexpected item: " + str(kind))
                if method == "item/completed" and kind == "agentMessage":
                    reply = params["item"]["text"]
            if method == "thread/tokenUsage/updated":
                usage = params.get("tokenUsage", {}).get("last")
            if method == "turn/completed":
                turn = params["turn"]
                if turn.get("status") != "completed" or reply is None:
                    raise AstraError("turn failed: " + str(turn.get("error")))
                info = {"latency_s": round(time.monotonic() - t0, 3), "usage": usage}
                try:
                    return json.loads(reply), info
                except json.JSONDecodeError as e:
                    raise AstraError("non-JSON reply: " + reply[:200]) from e

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
