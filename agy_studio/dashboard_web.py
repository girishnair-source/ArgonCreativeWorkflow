"""Local web view of the dashboard: ``python -m agy_studio --project <ID> dashboard serve [--open]``.

* Binds to 127.0.0.1 only, read-only, no auth; same Starlette/uvicorn stack the set3d viewer uses.
* ``/`` — single page that fetches ``/api/dashboard`` every 10 s and renders the panels client-side.
* ``/api/dashboard`` — the ``collect_dashboard`` JSON.
* ``/file/<project-relative path>`` — images / sheets / prompts from INSIDE the production folder only
  (``fsio.safe_join`` refuses anything else).
"""

from __future__ import annotations

import json
import threading
import webbrowser
from typing import Any

from .dashboard import collect_dashboard
from .fsio import PathSecurityError, safe_join
from .project import Project

_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".md", ".txt", ".json", ".yaml", ".yml", ".mp4", ".wav"}

INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>AGY Studio dashboard</title>
<style>
 body{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:14px 20px;background:#161a22;border-bottom:1px solid #262c38;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
 header h1{font-size:18px;margin:0} header .meta{color:#9aa4b2} header .pulse{margin-left:auto;color:#9aa4b2}
 main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;padding:14px 20px}
 section{background:#161a22;border:1px solid #262c38;border-radius:10px;padding:12px 14px;min-height:80px}
 section h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#9aa4b2;margin:0 0 8px}
 table{width:100%;border-collapse:collapse} td,th{padding:4px 6px;border-bottom:1px solid #222733;text-align:left;vertical-align:top}
 th{color:#9aa4b2;font-weight:600} code{background:#0f1115;padding:1px 5px;border-radius:4px;color:#c5d1ff}
 .error{color:#ff6b6b} .warn{color:#ffb454} .info{color:#7cc4ff} .ok{color:#6ddc8a}
 .banner{grid-column:1/-1;background:#3a1518;border:1px solid #ff6b6b;color:#ffd7d7;border-radius:10px;padding:10px 14px;white-space:pre-wrap}
 details{margin:6px 0} summary{cursor:pointer} pre{white-space:pre-wrap;background:#0f1115;padding:8px;border-radius:6px;max-height:260px;overflow:auto}
 img.thumb{max-width:100%;border-radius:6px;border:1px solid #262c38;margin-top:6px}
 .kpi{display:flex;gap:18px;flex-wrap:wrap} .kpi div b{display:block;font-size:20px}
 .wide{grid-column:1/-1}
</style></head><body>
<header><h1 id="title">AGY Studio</h1><span class="meta" id="meta"></span><span class="pulse" id="pulse">connecting…</span></header>
<main id="main"></main>
<script>
const fmt=s=>{s=Number(s||0);if(s<90)return s.toFixed(0)+'s';if(s<5400)return (s/60).toFixed(0)+' min';return (s/3600).toFixed(1)+' h'};
const esc=x=>String(x??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const file=p=>'/file/'+encodeURI(p);
function render(d){
  const p=d.project;
  document.getElementById('title').textContent='AGY Studio — '+p.id+' · '+p.title;
  document.getElementById('meta').textContent=`mode ${p.control_mode} · ${p.content_format} ${p.aspect_ratio} · lang ${p.language} · ${p.takes_per_clip} take(s)/clip · last engine activity ${d.status.last_activity_ago}`;
  const errs=d.risks.filter(r=>r.level==='error');
  let h='';
  if(errs.length) h+=`<div class="banner">NEEDS YOU\n${errs.map(r=>'• '+esc(r.text)).join('\n')}</div>`;
  // status
  h+='<section class="wide"><h2>Status</h2><table><tr><th>Scene</th><th>Stage</th><th>Waiting on you</th><th>Blocked by</th><th>ETA (engine time left)</th></tr>';
  const sc=d.status.scenes; const ids=Object.keys(sc);
  if(!ids.length) h+='<tr><td colspan=5>no scene started — run <code>ingest</code> → <code>g1</code></td></tr>';
  for(const id of ids){const s=sc[id];const e=(d.eta.per_scene||{})[id]||{};const hw=(e.human_gates_ahead||[]).length;
    h+=`<tr><td>${id}</td><td><code>${esc(s.stage)}</code></td><td>${esc(s.gates_waiting.join(', ')||'—')}</td><td>${esc(s.blocked_by.join(', ')||'—')}</td><td>${fmt(e.seconds)}${hw?' + '+hw+' human gate(s)':''}</td></tr>`;}
  h+='</table>';
  if(d.status.open_questions.length){h+='<p><b>Open questions (engine waiting):</b></p><ul>'+d.status.open_questions.map(q=>`<li><code>${esc(q.id)}</code> [${esc(q.scene_id||'GLOBAL')}] ${esc((q.text||'').slice(0,220))}</li>`).join('')+'</ul>';}
  h+='</section>';
  // risks
  h+='<section><h2>Things that may go wrong / need you</h2>'+(d.risks.length?'<ul>'+d.risks.map(r=>`<li class="${r.level}">${esc(r.text)}</li>`).join('')+'</ul>':'<p class="ok">nothing flagged</p>')+'</section>';
  // time/tokens/cost
  const lu=d.tokens.llm,vu=d.tokens.video;
  h+=`<section><h2>Time · tokens · cost</h2><div class="kpi">
   <div><b>${fmt(d.time.llm_total_s)}</b>in LLM calls (${lu.calls})</div>
   <div><b>${(lu.total_tokens||0).toLocaleString()}</b>LLM tokens</div>
   <div><b>${(vu.usage_total_tokens||0).toLocaleString()}</b>video tokens</div>
   <div><b>≈ ${Number(d.cost.spent_total||0).toFixed(2)}</b>spent${d.cost.prices_are_estimates?' (est.)':''}</div>
   <div><b>${fmt(d.eta.total_s)}</b>engine time left (${esc(d.eta.basis)})</div></div>`;
  const roles=Object.entries(d.time.per_role||{}).sort((a,b)=>b[1].total_s-a[1].total_s);
  if(roles.length){h+='<table><tr><th>Agent role</th><th>calls</th><th>total</th><th>slowest</th><th>err</th></tr>'+roles.map(([r,v])=>`<tr><td>${esc(r)}</td><td>${v.n}</td><td>${fmt(v.total_s)}</td><td>${fmt(v.max_s)}</td><td>${v.errors}</td></tr>`).join('')+'</table>';}
  if((d.time.gaps||[]).length){h+='<p class="warn">Silent gaps: '+d.time.gaps.slice(0,3).map(g=>fmt(g[0])+" after '"+esc(g[2].slice(0,50))+"'").join('; ')+'</p>';}
  h+='</section>';
  // bibles
  const b=d.bibles;
  h+=`<section><h2>Bibles & policy in use</h2><ul>
   <li>Global Context Bible: ${b.global_context_bible?'<code>'+esc(b.global_context_bible)+'</code>':'— (not built yet)'}</li>
   <li>Locked characters: ${esc(b.characters_locked.join(', ')||'none')}</li>
   <li>Voice specs: ${esc(b.voice_specs_locked.join(', ')||'none')}</li>
   <li>Video engines: primary <b>${esc(b.model_policy.video_primary)}</b>, secondary <b>${esc(b.model_policy.video_secondary)}</b></li>`;
  for(const [sid,s] of Object.entries(b.scenes||{})) h+=`<li>${sid}: ${s.scene_bible?'<code>'+esc(s.scene_bible)+'</code>':'—'} · ws <code>${esc(s.working_set_hash||'—')}</code> · env ${esc(s.environment||'—')} · cast ${esc((s.characters||[]).join(', ')||'—')}</li>`;
  h+='</ul></section>';
  // outputs
  const o=d.outputs;
  h+=`<section><h2>Outputs</h2><ul><li>Images in library: ${o.images_in_library} · storyboard frames: ${o.storyboard_frames}</li>
   <li>Takes: ${o.takes_completed} completed / ${o.takes_total} · selected: ${esc(o.takes_selected.join(', ')||'none yet')}</li>`;
  for(const t of o.takes_in_flight) h+=`<li>⏳ ${esc(t.take_id)} on ${esc(t.provider_model)} (${esc(t.dispatch_status)})</li>`;
  for(const [sid,tl] of Object.entries(o.handoffs||{})) if(tl) h+=`<li>${sid}: timeline <b>${esc(tl)}</b> delivered to Resolve</li>`;
  for(const m of o.final_mp4s||[]) h+=`<li>final: <a href="${file(m)}">${esc(m)}</a></li>`;
  h+='</ul></section>';
  // QA failures
  h+='<section class="wide"><h2>Takes that failed QA — prompt & references</h2>';
  if(!d.qa_failures.length) h+='<p class="ok">none</p>';
  for(const t of d.qa_failures.slice(-12)){const qa=t.qa||{};const why=(qa.hard_reject_reasons||[]).join(', ')||(t.error&&t.error.message)||t.dispatch_status;
    h+=`<details><summary><b>${esc(t.take_id)}</b> (${esc(t.provider_model||t.model)}) — <span class="error">${esc(why)}</span></summary>`;
    if(t.qa) h+=`<p>composite ${qa.composite} · ${esc(qa.recommendation)} · transcript ${qa.transcript_match} · adherence ${qa.prompt_adherence} · flags ${esc((qa.artifact_flags||[]).join(', ')||'—')}</p>`;
    if(t.prompt) h+=`<pre>${esc(t.prompt)}</pre>`;
    if(t.negative_prompt) h+=`<p>negative: <code>${esc(t.negative_prompt)}</code></p>`;
    const refs=[t.first_frame,t.last_frame,...(t.reference_images||[])].filter(Boolean);
    if(refs.length) h+='<p>references: '+refs.map(r=>`<a href="${file(r)}" target="_blank">${esc(r.split('/').pop())}</a>`).join(' · ')+'</p>';
    for(const s of t.contact_sheets||[]) h+=`<img class="thumb" src="${file(s)}" alt="${esc(s)}">`;
    h+='</details>';}
  h+='</section>';
  // decisions
  h+='<section class="wide"><h2>Latest decisions</h2>'+(d.decisions.length?'<ul>'+d.decisions.slice(-10).map(x=>`<li><code>${esc(x.id)}</code> ${esc(x.action)} ${esc(x.gate||'')} ${esc(x.target||'')} by ${esc(x.user)}${x.source?' ('+esc(x.source)+')':''}${x.rationale?' — '+esc(x.rationale):''}</li>`).join('')+'</ul>':'<p>none yet</p>')+'</section>';
  document.getElementById('main').innerHTML=h;
  document.getElementById('pulse').textContent='updated '+new Date().toLocaleTimeString()+' · refreshes every 10 s';
}
async function tick(){try{const r=await fetch('/api/dashboard',{cache:'no-store'});render(await r.json());}catch(e){document.getElementById('pulse').textContent='dashboard server unreachable';}}
tick();setInterval(tick,10000);
</script></body></html>
"""


def build_app(project: Project) -> Any:
    from starlette.applications import Starlette  # noqa: PLC0415
    from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    async def index(_req: Any) -> Any:
        return HTMLResponse(INDEX_HTML)

    async def api(_req: Any) -> Any:
        return JSONResponse(json.loads(json.dumps(collect_dashboard(project), default=str)))

    async def file(req: Any) -> Any:
        rel = req.path_params["path"]
        try:
            path = safe_join(project.root, rel)
        except (PathSecurityError, ValueError):
            return PlainTextResponse("forbidden", status_code=403)
        if not path.is_file() or path.suffix.lower() not in _ALLOWED_SUFFIXES:
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(str(path))

    return Starlette(routes=[Route("/", index), Route("/api/dashboard", api), Route("/file/{path:path}", file)])


def serve(project: Project, *, host: str = "127.0.0.1", port: int = 8766, open_browser: bool = False) -> None:
    import uvicorn  # noqa: PLC0415

    url = f"http://{host}:{port}/"
    print(f"AGY Studio dashboard for {project.project_id}: {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(build_app(project), host=host, port=port, log_level="warning")
