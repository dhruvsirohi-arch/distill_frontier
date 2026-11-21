"""Build a step-through viewer of one Phase 2 episode: for every Astra call, the exact head-camera
and lidar images it was shown, what it wrote it saw / inferred / decided, the command, the
server's reasoning summaries and token counts, and when the command took effect relative to the
sticks' releases.

    .venv/bin/python scripts/build_ringsim2_viewer.py outputs/ringsim2/astra_standard.jsonl 50000
Writes docs/ringsim/phase2/<mode>-<seed>/index.html and its frames/.
"""
from __future__ import annotations

import gzip
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def build(jsonl, seed, out_root=ROOT / "docs" / "ringsim" / "phase2"):
    from ringsim.p2.replay import replay
    rec = next(json.loads(l) for l in open(jsonl) if json.loads(l)["seed"] == seed)
    mode = rec["mode"]
    out = Path(out_root) / f"{mode}-{seed}"
    frames = out / "frames"
    res = replay(rec, frames_dir=frames)
    reproduced = [e["outcome"] for e in res["events"]] == [e["outcome"] for e in rec["events"]]
    trace_path = Path(jsonl).parent / "traces" / mode / f"{seed}.jsonl.gz"
    trace = {}
    if trace_path.exists():
        with gzip.open(trace_path, "rt") as f:
            for line in f:
                r = json.loads(line)
                trace.setdefault(r["call"], r)      # first attempt row per call
    calls = []
    for d in rec["decisions"]:
        tr = trace.get(d["call"], {})
        summ = [s for it in tr.get("reasoning_items", []) for s in it.get("summary", [])]
        raw = [c for it in tr.get("reasoning_items", []) for c in it.get("content", [])]
        u = d.get("usage") or {}
        calls.append(dict(
            call=d["call"], frame=d["frame"], t_obs=d["t_obs"], t_applied=d.get("t_applied"),
            applied=bool(d.get("applied", mode == "standard" and d.get("action") is not None)),
            superseded=bool(d.get("superseded")), after_end=bool(d.get("arrived_after_end")),
            latency=d.get("latency_s"), reasoning=d.get("reasoning") or {},
            action=d.get("action"), notes=d.get("notes", ""), error=d.get("error"),
            summaries=summ, raw_reasoning=raw, reasoning_tokens=u.get("reasoningOutputTokens"),
            out_tokens=u.get("outputTokens"), in_tokens=u.get("inputTokens"),
            img=f"frames/{d['frame']:04d}.jpg"))
    events = [dict(slot=e["slot"], release=e["release_t"], outcome=e["outcome"],
                   outcome_t=e["outcome_t"], hand=e["hand"]) for e in rec["events"]]
    end_t = max([e["outcome_t"] or e["release"] for e in events] +
                [c["t_obs"] for c in calls] + [1.0]) + 0.5
    data = dict(mode=mode, seed=seed, caught=rec["caught"], drops=rec["drops"], calls=calls,
                events=events, end_t=end_t, reproduced=reproduced)
    title = f"Astra trace · {mode} · seed {seed}"
    page = TEMPLATE.replace("%TITLE%", html.escape(title)).replace(
        "%DATA%", json.dumps(data).replace("</", "<\\/"))
    (out / "index.html").write_text(page)
    print(out / "index.html", f"{len(calls)} calls, reproduced={reproduced}")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#0a0b0e;--card:#13151a;--card2:#181b21;--line:rgba(255,255,255,.08);--ink:#f2f1ee;
--ink2:#b4b7bf;--ink3:#7c818c;--acc:#f2c230;--ok:#199e70;--bad:#d95926;--s1:#3987e5;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace;--sans:ui-sans-serif,system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--ink3);font:13px var(--mono);margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
#tl{width:100%;height:150px;display:block;cursor:pointer}
.ctl{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}
button{background:var(--card2);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font:13px var(--mono);cursor:pointer}
input[type=range]{flex:1;min-width:160px}
.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:14px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
img.view{width:100%;border-radius:8px;display:block;background:#000}
.k{font:12px var(--mono);color:var(--ink3);text-transform:uppercase;letter-spacing:.06em;margin:10px 0 2px}
.v{white-space:pre-wrap}.mono{font:12.5px/1.5 var(--mono);color:var(--ink2);white-space:pre-wrap;word-break:break-word}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font:12px var(--mono);border:1px solid var(--line);margin-right:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.acc{color:var(--acc)}
.legend{font:12px var(--mono);color:var(--ink3);margin-top:6px}
</style></head><body><div class="wrap">
<h1 id="h"></h1><div class="sub" id="sub"></div>
<div class="card"><svg id="tl" viewBox="0 0 1000 150" preserveAspectRatio="none"></svg>
<div class="legend">Pink ticks: stick releases (green = caught, red = missed). Blue bars: each call, from the frame it saw to when its command took effect (grey = never applied). Click to jump.</div></div>
<div class="ctl"><button id="prev">◀ prev</button><input type="range" id="sl" min="0" value="0"><button id="next">next ▶</button><span class="mono" id="pos"></span></div>
<div class="grid">
 <div class="card"><img class="view" id="img" alt="Head camera (two panels) and lidar views as shown to Astra">
  <div class="legend">Exactly what Astra was shown at this call: head RGB (left/right panels) and the lidar panorama/top-down views. (It also received depth, a 0.1/0.2 s clip and the point list.)</div></div>
 <div class="card" id="det"></div>
</div></div>
<script>
const D=%DATA%;const C=D.calls;let i=0;
const $=id=>document.getElementById(id);
$('h').textContent=`Astra trace — ${D.mode}, seed ${D.seed}`;
$('sub').textContent=`${D.caught}/${D.drops} caught · ${C.length} calls · replay ${D.reproduced?'reproduced':'MISMATCH'}`;
const X=t=>20+960*t/D.end_t;
function tl(){let s='';for(let t=0;t<=D.end_t;t+=1){s+=`<line x1="${X(t)}" x2="${X(t)}" y1="0" y2="150" stroke="rgba(255,255,255,.06)"/><text x="${X(t)+2}" y="146" fill="#7c818c" font-size="10">${t}s</text>`}
for(const e of D.events){const c=e.outcome==='caught'?'#199e70':'#d95926';s+=`<line x1="${X(e.release)}" x2="${X(e.release)}" y1="4" y2="36" stroke="#ff5fa2" stroke-width="2"/><rect x="${X(e.release)}" y="30" width="${Math.max(2,X(e.outcome_t||e.release)-X(e.release))}" height="6" fill="${c}"/>`}
const n=C.length,row=Math.min(14,100/Math.max(1,Math.ceil(n/Math.max(1,n))));
C.forEach((c,k)=>{const y=44+(k%24)*4;const x1=X(c.t_obs),x2=X(c.t_applied??Math.min(D.end_t,c.t_obs+(c.latency||0)));
s+=`<rect data-k="${k}" x="${x1}" y="${y}" width="${Math.max(1.5,x2-x1)}" height="3" fill="${c.applied?'#3987e5':'#5a5e66'}"/>`});
s+=`<line id="cur" x1="0" x2="0" y1="0" y2="140" stroke="#f2c230" stroke-width="1.5"/>`;$('tl').innerHTML=s}
function fmtA(a){if(!a)return'(none)';const f=h=>`[${h.pos.map(v=>v.toFixed(2)).join(', ')}] grip ${h.grip.toFixed(2)}`;return `left  ${f(a.left)}\nright ${f(a.right)}\nbase [${a.base.map(v=>v.toFixed(2))}] torso ${a.torso_yaw.toFixed(2)}`}
function esc(s){return String(s??'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}
function show(k){i=Math.max(0,Math.min(C.length-1,k));const c=C[i];$('sl').value=i;$('img').src=c.img;
$('pos').textContent=`call ${i+1}/${C.length} · saw t=${c.t_obs.toFixed(3)} s (frame ${c.frame})`;
const fate=c.applied?`<span class="pill ok">applied at t=${(c.t_applied??c.t_obs).toFixed(3)} s</span>`:c.superseded?'<span class="pill acc">superseded</span>':c.after_end?'<span class="pill bad">arrived after game end</span>':c.error?'<span class="pill bad">error</span>':'<span class="pill bad">never applied</span>';
const rec=D.events.filter(e=>e.release<=c.t_obs+1e-9).map(e=>`slot ${e.slot} released ${e.release.toFixed(2)} s → ${e.outcome}`).join('\n')||'none yet';
$('det').innerHTML=`${fate}<span class="pill">latency ${c.latency?.toFixed(2)??'—'} s</span><span class="pill">reasoning tokens ${c.reasoning_tokens??'—'}</span>
<div class="k">Observation (Astra)</div><div class="v">${esc(c.reasoning.observation)}</div>
<div class="k">Inference (Astra)</div><div class="v">${esc(c.reasoning.inference)}</div>
<div class="k">Decision (Astra)</div><div class="v">${esc(c.reasoning.decision)}</div>
<div class="k">Command</div><div class="mono">${esc(fmtA(c.action))}</div>
<div class="k">Notes to self</div><div class="mono">${esc(c.notes)}</div>
<div class="k">Server reasoning summaries</div><div class="mono">${c.summaries.length?esc(c.summaries.join('\n')):'(none sent: the model produced no hidden reasoning tokens on this call)'}</div>
${c.raw_reasoning.length?`<div class="k">Raw reasoning</div><div class="mono">${esc(c.raw_reasoning.join('\n'))}</div>`:''}
${c.error?`<div class="k">Error</div><div class="mono bad">${esc(c.error)}</div>`:''}
<div class="k">Ground truth at the frame it saw (not shown to Astra)</div><div class="mono">${esc(rec)}</div>`;
$('cur').setAttribute('x1',X(c.t_obs));$('cur').setAttribute('x2',X(c.t_obs))}
tl();$('sl').max=C.length-1;$('sl').oninput=e=>show(+e.target.value);$('prev').onclick=()=>show(i-1);$('next').onclick=()=>show(i+1);
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(i-1);if(e.key==='ArrowRight')show(i+1)});
$('tl').onclick=e=>{const r=$('tl').getBoundingClientRect();const t=((e.clientX-r.left)/r.width*1000-20)/960*D.end_t;let b=0,bd=1e9;C.forEach((c,k)=>{const dd=Math.abs(c.t_obs-t);if(dd<bd){bd=dd;b=k}});show(b)};
show(0);
</script></body></html>
"""

if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]))
