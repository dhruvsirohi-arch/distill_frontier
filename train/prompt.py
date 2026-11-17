"""Chat format shared by SFT and evaluation (single source of truth for prompts and parsing)."""
from __future__ import annotations

import json
import math

SYSTEM_PROMPT = (
    "Control an SO-101 arm: pick the bowl off the floor, place it upright inside the table target "
    "region, open, lift clear, then done. Units mm, yaw deg; gripper points down. Reply with one "
    'JSON action: {"action":"move_to","x":X,"y":Y,"z":Z} | {"action":"open_gripper"} | '
    '{"action":"close_gripper"} | {"action":"done"}'
)
# The model-facing JSON uses integer millimetres (Qwen tokenizes digits one by one, so "88" is 2
# tokens vs 5 for "0.088"). Lossless: the expert already rounds targets to 1 mm. The env, raw data
# and every file outside this module stay in meters.

_C = (",", ":")


def _mm(v):
    return int(round(float(v) * 1000))


def obs_to_model(obs: dict) -> dict:
    b, t, e = obs["bowl"], obs["target"], obs["ee"]
    return {"step": obs["step"],
            "bowl": {"x": _mm(b["x"]), "y": _mm(b["y"]), "z": _mm(b["z"]), "yaw": int(b["yaw"])},
            "target": {"x": _mm(t["x"]), "y": _mm(t["y"]), "z": _mm(t["z"]), "sx": _mm(t["sx"]), "sy": _mm(t["sy"])},
            "ee": {"x": _mm(e["x"]), "y": _mm(e["y"]), "z": _mm(e["z"])},
            "gripper": obs["gripper"], "grasped": obs["grasped"], "last_result": obs["last_result"]}


def action_to_text(a: dict | None) -> str:
    if a is None:
        return '{"action":"invalid"}'
    if a["action"] == "move_to":
        return json.dumps({"action": "move_to", "x": _mm(a["x"]), "y": _mm(a["y"]), "z": _mm(a["z"])}, separators=_C)
    return json.dumps({"action": a["action"]}, separators=_C)


def user_text(obs: dict, history: list[dict]) -> str:
    hist = "[" + ",".join(action_to_text(h if h.get("action") != "invalid" else None) for h in history) + "]"
    return f"obs:{json.dumps(obs_to_model(obs), separators=_C)}\nlast:{hist}"


def messages(obs: dict, history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text(obs, history)}]


def prompt_text(tokenizer, obs: dict, history: list[dict]) -> str:
    kw = {"enable_thinking": False}  # Qwen3: disable thinking; ignored by other templates
    return tokenizer.apply_chat_template(messages(obs, history), tokenize=False, add_generation_prompt=True, **kw)


def completion_text(tokenizer, action: dict) -> str:
    """Assistant target = action JSON + end-of-turn token."""
    eot = "<|im_end|>"  # Qwen chat format
    return action_to_text(action) + eot


def parse_action(text: str) -> dict | None:
    """Strict schema validation. Returns None for anything invalid."""
    t = text.strip()
    if t.endswith("<|im_end|>"):
        t = t[: -len("<|im_end|>")].strip()
    try:
        a = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(a, dict) or "action" not in a:
        return None
    k = a["action"]
    if k in ("open_gripper", "close_gripper", "done"):
        return {"action": k} if set(a) == {"action"} else None
    if k == "move_to":
        if set(a) != {"action", "x", "y", "z"}:
            return None
        try:
            v = [float(a[c]) for c in "xyz"]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(x) and abs(x) < 2000.0 for x in v):
            return None
        return {"action": "move_to", "x": v[0] / 1000.0, "y": v[1] / 1000.0, "z": v[2] / 1000.0}
    return None
