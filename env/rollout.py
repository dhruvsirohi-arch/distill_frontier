"""Shared episode runner used by the expert check, data generation, training checkpoints and eval."""
from __future__ import annotations

import time

import numpy as np


def run_episode(env, policy, seed: int, episode_id: str = "", history_len: int = 2) -> dict:
    t0 = time.time()
    obs = env.reset(seed, episode_id=episode_id, policy_name=policy.name)
    params = policy.reset(seed, obs)
    history: list[dict] = []
    steps = []
    done, info = False, {}
    while not done:
        out = policy.act(obs, history[-history_len:] if history_len else [])
        action, raw = (out if isinstance(out, tuple) else (out, None))
        prev_obs = obs
        obs, done, info = env.step(action)
        rec = {"step": prev_obs["step"], "observation": prev_obs,
               "history": [dict(h) for h in history[-history_len:]] if history_len else [],
               "action": action, "result": info["result"]}
        if raw is not None:
            rec["raw_output"] = raw
        steps.append(rec)
        history.append(action if action is not None else {"action": "invalid"})
    st = env.stats
    tg = env.ecfg["target"]
    return {
        "episode_id": episode_id or f"seed{seed}",
        "seed": int(seed),
        "policy": policy.name,
        "spawn": env.spawn,
        "variation": params,
        "steps": steps,
        "n_steps": len(steps),
        "success": bool(st.success),
        "failure_type": st.failure_type,
        "success_detail": st.success_detail,
        "metrics": {
            "path_length": round(st.path_length, 4),
            "n_collisions": st.n_collisions,
            "n_rejected": st.n_rejected,
            "n_failed_grasps": st.n_failed_grasps,
            "n_invalid": st.n_invalid,
            "dropped": st.dropped,
            "place_margin": round(float(st.success_detail.get("margin", float("nan"))), 4),
            "final_bowl": [round(float(v), 4) for v in env.bowl_pose()[0]],
        },
        "wall_time": round(time.time() - t0, 3),
    }


def run_episodes_lockstep(envs, policy, seeds, episode_ids, history_len: int = 2) -> list[dict]:
    """Run len(envs) episodes in lockstep so a batched policy (Qwen) does one generate per step.

    Each env runs exactly the same per-episode logic as run_episode; only decision calls are batched.
    """
    n = len(envs)
    t0 = time.time()
    obs = [env.reset(s, episode_id=eid, policy_name=policy.name) for env, s, eid in zip(envs, seeds, episode_ids)]
    params = [policy.reset(s, o) for s, o in zip(seeds, obs)]
    hist = [[] for _ in range(n)]
    steps = [[] for _ in range(n)]
    done = [False] * n
    while not all(done):
        idx = [i for i in range(n) if not done[i]]
        outs = policy.act_batch([obs[i] for i in idx], [hist[i][-history_len:] for i in idx])
        for i, (action, raw) in zip(idx, outs):
            prev = obs[i]
            obs[i], d, info = envs[i].step(action)
            steps[i].append({"step": prev["step"], "observation": prev,
                             "history": [dict(h) for h in hist[i][-history_len:]],
                             "action": action, "result": info["result"], "raw_output": raw})
            hist[i].append(action if action is not None else {"action": "invalid"})
            done[i] = d
    res = []
    for i, env in enumerate(envs):
        st = env.stats
        res.append({
            "episode_id": episode_ids[i], "seed": int(seeds[i]), "policy": policy.name, "spawn": env.spawn,
            "variation": params[i], "steps": steps[i], "n_steps": len(steps[i]), "success": bool(st.success),
            "failure_type": st.failure_type, "success_detail": st.success_detail,
            "metrics": {"path_length": round(st.path_length, 4), "n_collisions": st.n_collisions,
                        "n_rejected": st.n_rejected, "n_failed_grasps": st.n_failed_grasps,
                        "n_invalid": st.n_invalid, "dropped": st.dropped,
                        "place_margin": round(float(st.success_detail.get("margin", float("nan"))), 4),
                        "final_bowl": [round(float(v), 4) for v in env.bowl_pose()[0]]},
            "wall_time": round(time.time() - t0, 3),
        })
    return res
