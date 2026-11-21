"""Fleet cost calculator and QuantConfig snippet generator (JavaScript).

View 2 derives concurrent-agent fleet economics from p13_ragged_concurrency.csv
(modeled); view 3 emits a QuantConfig snippet that tests/test_explorer.py
compile-checks against the real dataclass, citing source CSVs and hashes.
"""

JS_VIEWS2 = r"""
/* ================= View 2: fleet cost calculator (p13) ================= */
function renderFleet(){
  var fig=$("fig-c"),leg=$("leg-c"),badges=$("badges-c"),h3=$("h3-c"),stats=$("fleet-stats");
  clearNode(leg);clearNode(stats);
  h3.innerHTML="Concurrent 20-turn agent sessions @ "+fmtCtx(S.ctx)+", "+S.budget+" GiB budget "
    +noteFor(fig,"fleet");
  badges.innerHTML=badge("modeled","modeled (analytical byte model)")
    +' <span style="color:var(--faint);font-size:11px">model: Qwen2.5-7B geometry; ITL modeled for Llama-2-7b (p7)</span>';
  var rows=D.fleet.filter(function(f){return f.budget===S.budget&&f.ctx===S.ctx;});
  if(!rows.length){
    emptyState(fig,"<b>not measured</b> — p13 fleet rows exist at 32K and 64K only. "
      +"Switch context; greyed chips are honest absences.");
    return;
  }
  var base=rows.filter(function(r){return r.config==="fp16";})[0];
  var cow=rows.filter(function(r){return r.config==="fp16_cow";})[0];
  var best=rows.reduce(function(a,r){return r.sessions>a.sessions?r:a;},rows[0]);
  var cowRatio=cow?cow.sessions/base.sessions:null;
  var allRatio=best.sessions/base.sessions;
  [["FP16 baseline",base.sessions+" sessions","no CoW, no quant — "+fmtGiB(base.used_b)+" GiB at max"],
   ["Best fit: "+best.config,best.sessions+" sessions",(allRatio).toFixed(1)+"× baseline ("+fmtGiB(best.used_b)+" GiB at max)"],
   ["CoW alone",cow?(cow.sessions+" sessions").concat(""):"—",
     cow?("fp16_cow vs fp16 = "+cowRatio.toFixed(2)+"×"):("n/a"),
     "quant on top of CoW = "+(best.sessions/cow.sessions).toFixed(1)+"× more"]
  ].forEach(function(card){
    var d=document.createElement("div");d.className="stat";
    d.innerHTML='<div class="v">'+esc(card[1])+'</div><div class="k">'+esc(card[0])
      +'</div><div class="d">'+esc(card[2])+(card[3]?' · '+esc(card[3]):"")+'</div>';
    stats.appendChild(d);
  });
  var stat4=document.createElement("div");stat4.className="stat";
  // p13's modeled_decode_tok_s is ALREADY fleet-wide across all sessions
  // (concurrency_sim.py: modeled_tok_s = sessions * 1e6 / step_time).
  // Multiplying by sessions again would inflate the CSV by 161x-543x.
  stat4.innerHTML='<div class="v">'+best.tok_s.toFixed(0)
    +'</div><div class="k">fleet decode tok/s</div><div class="d">'
    +"p13 modeled — total across all "+best.sessions+" sessions (do not multiply by session count)</div>";
  stats.appendChild(stat4);
  clearNode(fig);
  var W=560,H=210,L=40,R=10,T=10,B=44;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var maxS=best.sessions*1.15;
  var y=lin(0,maxS,H-B,T);
  var band=(W-L-R)/rows.length;
  rows.forEach(function(r,i){
    var cx=L+band*(i+0.5);
    el("rect",{x:cx-band*0.36,y:y(r.sessions),width:band*0.72,height:H-B-y(r.sessions),
      class:"bar l"+(r.cow?(r.config==="fp16_cow"?"1":(r.config==="k4v4_hadamard_cow"?"0":"2")):"3"),rx:3},svg);
    el("text",{x:cx,y:y(r.sessions)-5,class:"ticktext","text-anchor":"middle"},svg)
      .textContent=r.sessions;
    el("text",{x:cx,y:H-24,class:"ticktext","text-anchor":"middle"},svg).textContent=r.config.replace("_cow"," +CoW");
    el("text",{x:cx,y:H-9,class:"ticktext","text-anchor":"middle",fill:"var(--faint)"},svg)
      .textContent=r.sessions/base.sessions<1.05?"baseline":(r.sessions/base.sessions).toFixed(1)+"×";
  });
  var k=document.createElement("span");k.className="k";
  k.innerHTML='sessions within '+S.budget+' GiB — <span style="color:var(--faint)">ratios vs fp16 no-CoW baseline; '
    +'CoW alone '+cowRatio.toFixed(2)+'×, quant+CoW '+allRatio.toFixed(1)+'×</span>';
  leg.appendChild(k);
  var cap=document.createElement("div");cap.style.cssText="color:var(--faint);font-size:11px;padding-top:4px";
  cap.textContent="Per-session bytes split (p13 bytes_per_token × ctx, shared prefix deduplicated by CoW):";
  fig.appendChild(cap);
  rows.forEach(function(r){
    var row=document.createElement("div");row.style.cssText="display:flex;align-items:center;gap:8px;margin-top:5px;font-size:11px;color:var(--muted)";
    var lab=document.createElement("span");lab.style.cssText="width:170px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    lab.textContent=r.config;
    var barwrap=document.createElement("span");
    barwrap.style.cssText="flex:1;height:10px;border-radius:5px;overflow:hidden;display:flex";
    var total=(r.shared_b?r.shared_b:0)+r.private_b;
    if(r.shared_b){
      var sh=document.createElement("i");sh.style.cssText="background:var(--teal);width:"+(100*r.shared_b/total)+"%";
      barwrap.appendChild(sh);
      var pr=document.createElement("i");pr.style.cssText="background:var(--amber);width:"+(100*r.private_b/total)+"%";
      barwrap.appendChild(pr);
    }else{
      var one=document.createElement("i");one.style.cssText="background:var(--red);width:100%";
      barwrap.appendChild(one);
    }
    row.appendChild(lab);row.appendChild(barwrap);
    var val=document.createElement("span");val.textContent=fmtMiB(total)+" MiB"+(r.shared_b?" (shared teal + private amber)":" (no CoW)");
    row.appendChild(val);
    fig.appendChild(row);
  });
}
/* ================= View 3: QuantConfig snippet ================= */
function sha12(file){
  var a=(D.meta.artifacts||[]).filter(function(x){return x.file===file;})[0];
  return a?a.sha256.slice(0,12):"";
}
function fleetRow(cfgName){
  return D.fleet.filter(function(f){return f.budget===S.budget&&f.ctx===S.ctx&&f.config===cfgName;})[0];
}
function recommend(){
  var r={k:8,v:4,scope:"none",conc:"k8v4_cow",mode:"v1",why:[]};
  if(S.task==="agent"){
    r={k:4,v:8,scope:"none",conc:null,mode:"agent",
      why:["p14 (simulated): K is a measured null — 6/6,400 route flips, all inside the 20K window",
           "V precision decides sessions: K4V8 resolves 100%, K8V4 only 71.9% @ 64K, same bit budget"]};
    return r;
  }
  if(S.phase==="v2"){
    if(S.budget<=24){
      r={k:4,v:4,scope:"block_diag",conc:"k4v4_hadamard_cow",mode:"density",
        why:["p10/p11 (projected): block-diagonal Hadamard rescues INT4-K — 0.8455 -> 0.9652 @ 32K",
             "max density: 4.0 bits/elem, 3.95x VRAM compression (p7_vram_crossval)"]};
    }else{
      r={k:8,v:4,scope:"block_diag",conc:"k8v4_cow",mode:"quality",
        why:["p11 (projected): head-adaptive tier — top 25% retrieval heads stay K8V4, streaming heads drop to K4V4",
             "quantized append pays +0.36 ms TTFT, amortizes in 0.06 decode steps (p7_latency)"]};
    }
    return r;
  }
  r.why=["p6 (v1.0): K8V4 within 0.22-0.36% of FP16 @64K, up to 0.43% rel @128K multihop",
         "p4: unrotated INT4-K is the AVOID row — the v2.0 toggle exists because of it"];
  return r;
}
function renderSnippet(){
  var pre=$("snip"),warn=$("snip-warn"),ok=$("snip-ok"),h3=$("h3-d"),badges=$("badges-d");
  var rec=recommend(),task=S.task;
  h3.innerHTML="QuantConfig for "+(TASKS.filter(function(t){return t[0]===task;})[0][1])
    +" @ "+fmtCtx(S.ctx)+" ("+S.budget+" GB) "+'<span style="color:var(--faint);text-transform:none;letter-spacing:0">one-click, cited, compile-checked</span>';
  badges.innerHTML=badge(rec.mode==="v1"?"replay-baseline":(rec.mode==="agent"?"simulated":"projected"),
    rec.mode==="v1"?"v1.0 baseline rows":(rec.mode==="agent"?"simulated harness":"v2.0 projections"));
  var fr=rec.conc?fleetRow(rec.conc):null;
  var lines=[
    "# kv-quant v2.0.0 explorer snippet — "+task+" @ "+fmtCtx(S.ctx)+", "+S.budget+" GiB budget",
    "# Every number below traces to a SHA256-verified CSV:",
    "#   p6_v1_frontier_sweep.csv       #"+sha12("p6_v1_frontier_sweep.csv")+"  (v1.0 frontier)",
    "#   p13_ragged_concurrency.csv     #"+sha12("p13_ragged_concurrency.csv")+"  (modeled fleet)",
    "#   p14_agent_trajectory_drift.csv #"+sha12("p14_agent_trajectory_drift.csv")+"  (simulated agent)",
    "from kv_quant.config import QuantConfig",
    "",
    "cfg = QuantConfig(",
    "    k_bits="+rec.k+",          # "+rec.why[0],
    "    v_bits="+rec.v+",          # "+rec.why[1]
  ];
  if(rec.scope==="block_diag"){
    lines.push("    hadamard_scope=\"block_diag\",   # rotary subspace only (rotary_dim=128)");
    lines.push("    rotary_dim=128,");
    lines.push("    # NOTE: apply the rotation via hadamard.rotate_qk(q, k) BEFORE quantizing");
    lines.push("    # keys. Do NOT set the legacy use_hadamard=True flag (known Q/K sign");
    lines.push("    # mismatch, DECISIONS.md ADR-012 / KERNEL_NOTES.md). Consequently");
    lines.push("    # cfg.resolve_hadamard_scope() reports 'none' on this config by design:");
    lines.push("    # the rotation is applied explicitly, not through that flag.");
  }
  lines.push("    k_scheme=\"sym\",          # keys: symmetric, routing-safe");
  lines.push("    v_scheme=\"asym\",         # values: asymmetric range");
  lines.push("    k_quant_stage=\"post_rope\",");
  lines.push("    group_size=64,          # matches HybridPagedKVCache sealing granularity");
  lines.push("    sink_tokens=4,          # FP16 anchors; removing them costs 0.96 pts @ 64K (p6)");
  lines.push(")");
  lines.push("# Serving-side knobs live on OTHER dataclasses (kv_quant.config):");
  lines.push("#   CacheConfig(device=\"cuda\", pin_memory=True)   # device placement");
  if(fr){
    lines.push("#   BenchConfig(concurrency="+fr.sessions+")              # p13: "+rec.conc
      +" fits "+fr.sessions+" sessions @ "+S.budget+" GiB, "+fmtCtx(S.ctx)+" ctx (modeled)");
  }else{
    lines.push("#   BenchConfig(concurrency=...)  # no p13 fleet row for this family; see fleet view");
  }
  pre.textContent=lines.join("\n");
  warn.innerHTML=(S.phase==="v1"&&S.ctx>=65536)
    ?"AVOID unrotated INT4-K here: at this context the v1.0 cliff config (k4v4 post-RoPE, "
    +"no rotation) loses 14.8 pts @ 32K and 21.8 pts @ 64K on needle (p4). Flip the phase "
    +"toggle to v2.0 to rotate keys first."
    :(rec.mode==="density"||rec.mode==="quality"
      ?"v2.0 rescue rows are <b>projections</b> from declared synthetic fixtures "
      +"(p10 micro-kv-2k, p11 bimodal-calibration) — not model forward passes. "
      +"Do not quote them as measurements (README §3.4)."
      :(rec.mode==="agent"
        ?"Agent rows are a <b>simulated</b> harness driven by measured quantizer error "
        +"— no LLM ran on the host (ADR-012)."
        :""));
  warn.style.display=warn.innerHTML?"block":"none";
  ok.textContent="Snippet is compile-checked and instantiated against the real QuantConfig "
    +"in CI (tests/test_explorer.py::test_snippet_compiles_against_real_config).";
  var btn=$("copy");
  btn.onclick=function(){
    navigator.clipboard.writeText(pre.textContent).then(function(){
      btn.textContent="copied ✓";setTimeout(function(){btn.textContent="copy";},1400);
    });
  };
}
"""
