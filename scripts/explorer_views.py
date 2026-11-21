"""View renderers (JavaScript) for docs/explorer.html.

Panel A: position-resolved accuracy / task metric (p4 needle, p5 ecological,
p14 agent bars).  Panel B: context-resolved accuracy + dlogP (p6 v1.0
frontier, p10/p11 v2.0 rescue projections, p14 agent resolution).  The fleet
calculator and QuantConfig snippet live in scripts/explorer_views2.py.
"""

JS_VIEWS = r"""
var depthTip = null;  // single reusable hover tooltip (no node leak on re-render)
/* ================= Panel A: position-resolved ================= */
function renderPanelA(){
  var fig=$("fig-a"),leg=$("leg-a"),h3=$("h3-a"),badges=$("badges-a");
  clearNode(leg);
  if(S.task==="agent"){h3.innerHTML="K-amnesia vs V-schema corruption per config "
    +noteFor(fig,"agent");badges.innerHTML=badge("simulated","simulated harness");
    renderAgentBars(fig,leg);return;}
  var curves=D.depth[S.task]||{};
  h3.innerHTML=TASKS.filter(function(t){return t[0]===S.task;})[0][1]
    +" — "+D.task_metric[S.task].label+" vs insertion depth @ "+fmtCtx(S.ctx)
    +noteFor(fig,"depth");
  var has=S.phase==="v2";
  badges.innerHTML=badge("replay-baseline",has?"v1.0 baseline (reference)":"v1.0 baseline")
    +(has?"<span style='color:var(--faint);font-size:11px'> — no depth-resolved v2.0 rows exist; rescue curves are context-resolved (Panel B)</span>":"");
  var configs=Object.keys(curves).sort();
  var ctxSeries=configs.filter(function(c){return curves[c][S.ctx];});
  if(!ctxSeries.length){
    emptyState(fig,"<b>not measured</b> — no depth-resolved rows for this task at "
      +fmtCtx(S.ctx)+" (greyed contexts are honest absences, not zeros)");
    return;
  }
  var maxPts=0;ctxSeries.forEach(function(c){maxPts=Math.max(maxPts,curves[c][S.ctx].length);});
  if(maxPts<2){
    emptyState(fig,"<b>not position-resolved</b> — p5 ecological rows for this task "
      +"measure a single insertion depth only, so no depth curve exists. "
      +"Panel B shows the context-resolved p6 frontier instead.");
    return;
  }
  var metricKey=D.task_metric[S.task].key;
  var vals=[];ctxSeries.forEach(function(c){curves[c][S.ctx].forEach(function(p){vals.push(p[1]);});});
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);
  var pad=(hi-lo)*0.08||0.02;lo=Math.max(0,lo-pad);hi=Math.min(1,hi+pad);
  if(D.task_metric[S.task].key==="fact_retention_pct"){hi=100;}
  drawDepthChart(fig,leg,ctxSeries,curves,lo,hi,metricKey);
}
function drawDepthChart(fig,leg,configs,curves,lo,hi,metricKey){
  clearNode(fig);
  var W=560,H=300,L=46,R=10,T=8,B=30;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var x=lin(0,100,L,W-R),y=lin(lo,hi,H-B,T);
  for(var i=0;i<=4;i++){
    var yv=lo+(hi-lo)*i/4,yy=y(yv);
    el("line",{x1:L,x2:W-R,y1:yy,y2:yy,class:"gridline"},svg);
    el("text",{x:L-6,y:yy+3,class:"ticktext","text-anchor":"end"},svg)
      .textContent=yv.toFixed(2);
  }
  [0,25,50,75,100].forEach(function(d){
    el("text",{x:x(d),y:H-8,class:"ticktext","text-anchor":"middle"},svg)
      .textContent=d+"%";
  });
  el("text",{x:L-34,y:T+6,class:"xlab"},svg).textContent=metricKey==="fact_retention_pct"?"%":"acc";
  el("text",{x:(L+W-R)/2,y:H-2,class:"xlab","text-anchor":"middle"},svg).textContent="insertion depth (% of context)";
  var cross=el("line",{y1:T,y2:H-B,class:"cross",visibility:"hidden"},svg);
  var tip=depthTip;
  if(!tip){
    tip=document.createElement("div");
    tip.style.cssText="position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--grid);padding:6px 9px;font-size:11px;border-radius:6px;display:none;z-index:99;color:var(--ink)";
    document.body.appendChild(tip);depthTip=tip;
  }
  configs.forEach(function(cfg,ci){
    var pts=curves[cfg][S.ctx].map(function(p){return [x(p[0]),y(p[1])];});
    el("path",{d:linePath(pts),class:"cl l"+(ci%6)},svg);
    var sw=document.createElement("span");sw.className="k";
    sw.innerHTML='<span class="sw" style="background:var(--line'+(ci%6)+')"></span>'+esc(cfg);
    leg.appendChild(sw);
  });
  svg.addEventListener("mousemove",function(ev){
    var rect=svg.getBoundingClientRect();
    var px=(ev.clientX-rect.left)*(W/rect.width);
    if(px<L||px>W-R){cross.setAttribute("visibility","hidden");tip.style.display="none";return;}
    var depth=(px-L)/(W-R-L)*100;
    cross.setAttribute("x1",px);cross.setAttribute("x2",px);cross.setAttribute("visibility","visible");
    var rows=configs.map(function(cfg,ci){
      var best=curves[cfg][S.ctx].reduce(function(a,p){return Math.abs(p[0]-depth)<Math.abs(a[0]-depth)?p:a;});
      return '<div><span class="sw" style="background:var(--line'+(ci%6)+');display:inline-block;width:12px;height:3px;margin-right:5px"></span>'
        +esc(cfg)+" <b>"+best[1].toFixed(3)+"</b></div>";
    }).join("");
    tip.innerHTML="<b>depth "+depth.toFixed(0)+"%</b>"+rows;
    tip.style.display="block";tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY+10)+"px";
  });
  svg.addEventListener("mouseleave",function(){cross.setAttribute("visibility","hidden");tip.style.display="none";});
}
/* ================= Panel B: context-resolved ================= */
function renderPanelB(){
  var fig=$("fig-b"),leg=$("leg-b"),h3=$("h3-b"),badges=$("badges-b");
  clearNode(leg);
  if(S.task==="agent"){h3.innerHTML="20-turn resolution vs loop collapse per config "
    +noteFor(fig,"agent");badges.innerHTML=badge("simulated","simulated harness");
    renderAgentResolved(fig,leg);return;}
  if(S.phase==="v2"){
    if(S.task!=="needle"){
      h3.innerHTML="v2.0 rescue curves "+noteFor(fig,"rescue");
      badges.innerHTML=badge("projected","projected");
      emptyState(fig,"<b>not measured</b> — v2.0 Hadamard / head-adaptive rescue rows exist "
        +"for <b>needle retrieval only</b> (p10/p11, declared synthetic fixtures). "
        +"See README §3.4: do not extrapolate beyond the stated fixture.");
      return;
    }
    h3.innerHTML="Cliff rescue — accuracy vs context (Wilson 95% CI bands) "
      +noteFor(fig,"rescue");
    badges.innerHTML=badge("projected","projected (replay adapter)")
      +' <span style="color:var(--faint);font-size:11px">p10 = micro-kv-2k fixture · p11 = bimodal-calibration fixture</span>';
    drawRescueChart(fig,leg);return;
  }
  h3.innerHTML="Accuracy (solid) &amp; ΔlogP (dashed) vs context — contexts within the "
    +S.budget+" GB envelope "+noteFor(fig,"frontier");
  badges.innerHTML=badge("replay-baseline","v1.0 baseline")
    +' <span style="color:var(--faint);font-size:11px">dotted right axis: ΔlogP sensor</span>';
  drawFrontierChart(fig,leg);
}
function ctxPositions(available){
  var pos={},step=1/(CTXS.length-1),idx={};
  CTXS.forEach(function(c,i){idx[c[0]]=i;});
  available.forEach(function(c){pos[c]=idx[c]*step;});
  return pos;
}
function frameGrid(fig,svg,W,H,L,R,T,B,ylo,yhi,ylab){
  var y=lin(ylo,yhi,H-B,T);
  for(var i=0;i<=4;i++){
    var yv=ylo+(yhi-ylo)*i/4,yy=y(yv);
    el("line",{x1:L,x2:W-R,y1:yy,y2:yy,class:"gridline"},svg);
    el("text",{x:L-6,y:yy+3,class:"ticktext","text-anchor":"end"},svg).textContent=yv.toFixed(2);
  }
  el("text",{x:L-38,y:T+4,class:"xlab"},svg).textContent=ylab;
  return y;
}
function drawFrontierChart(fig,leg){
  clearNode(fig);
  var front=D.frontier[S.task]||{},budgetMiB=S.budget*1024;
  var fit=Object.keys(front).sort().filter(function(c){
    return Object.keys(front[c]).some(function(ctx){return front[c][ctx].vram_mib<=budgetMiB;});
  });
  if(!fit.length){
    emptyState(fig,"<b>nothing fits</b> — no v1.0 config's theoretical VRAM "
      +"(p6 <code>theoretical_vram_mib</code>) fits "+S.budget+" GB for this task at any "
      +"context. Raise the budget, or see the fleet view: paged CoW sharing is how "
      +"multiple sessions fit at all (p13).");
    return;
  }
  var W=560,H=300,L=52,R=44,T=8,B=30;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var pos=ctxPositions(CTXS.map(function(c){return c[0];}));
  var x=function(c){return 52+pos[c]*(W-L-R);};
  var ylo=1,yhi=0,dlo=1e9,dhi=-1e9;
  fit.forEach(function(c){Object.keys(front[c]).forEach(function(ctx){
    if(front[c][ctx].vram_mib>budgetMiB){return;}
    ylo=Math.min(ylo,front[c][ctx].acc);yhi=Math.max(yhi,front[c][ctx].acc);
    dlo=Math.min(dlo,front[c][ctx].dlogp);dhi=Math.max(dhi,front[c][ctx].dlogp);
  });});
  ylo=Math.max(0,Math.floor((ylo-0.02)*20)/20);yhi=Math.min(1,Math.ceil((yhi+0.02)*20)/20);
  var y=frameGrid(fig,svg,W,H,L,R,T,B,ylo,yhi,"acc");
  var yr=lin(Math.max(0,dlo-0.05),dhi+0.05,H-B,T);
  CTXS.forEach(function(c){
    el("text",{x:x(c[0]),y:H-8,class:"ticktext","text-anchor":"middle"},svg).textContent=c[1];
  });
  el("text",{x:W-6,y:T+4,class:"xlab","text-anchor":"end"},svg).textContent="ΔlogP";
  fit.forEach(function(cfg,ci){
    var pts=[],dpts=[];
    CTXS.forEach(function(c){
      var st=front[cfg][c[0]];
      if(st&&st.vram_mib<=budgetMiB){pts.push([x(c[0]),y(st.acc)]);dpts.push([x(c[0]),yr(st.dlogp)]);}
    });
    if(pts.length<2){return;}
    el("path",{d:linePath(pts),class:"cl l"+(ci%6)},svg);
    el("path",{d:linePath(dpts),class:"cl l"+(ci%6),"stroke-dasharray":"5 4",opacity:.55},svg);
    pts.forEach(function(p){el("circle",{cx:p[0],cy:p[1],r:2.2,class:"l"+(ci%6)},svg);});
    var sw=document.createElement("span");sw.className="k";
    sw.innerHTML='<span class="sw" style="background:var(--line'+(ci%6)+')"></span>'+esc(cfg)
      +(cfg.indexOf("k16v16")===0?" (FP16 baseline)":"");
    leg.appendChild(sw);
  });
  fig.appendChild(document.createTextNode(""));
  var cap=document.createElement("div");cap.style.cssText="color:var(--faint);font-size:11px;padding-top:4px";
  cap.textContent="solid = acc_mean, dashed = ΔlogP (right axis). VRAM gate uses p6 "
    +"theoretical_vram_mib, which is a CONTEXT-LENGTH ENVELOPE (identical for every config at a "
    +"given context in this sweep), not a per-config quantization filter: rows past the budget "
    +"are dropped for all configs alike.";
  fig.appendChild(cap);
}
function drawRescueChart(fig,leg){
  clearNode(fig);
  var order=["fp16","k4v4_postrope","k4v4_hadamard","head_adaptive_4p5_hadamard","k4v4_hadamard_projected"];
  var cfgs=order.filter(function(c){return D.rescue[c];});
  var W=560,H=300,L=52,R=44,T=8,B=30;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var pos=ctxPositions([4096,16384,32768,65536,131072]);
  var x=function(c){return L+pos[c]*(W-L-R);};
  var ylo=1,yhi=0;
  cfgs.forEach(function(c){Object.keys(D.rescue[c]).forEach(function(ctx){
    var st=D.rescue[c][ctx];if(st.acc===null){return;}
    ylo=Math.min(ylo,st.acc);yhi=Math.max(yhi,st.acc);
  });});
  ylo=Math.max(0,Math.floor((ylo-0.02)*20)/20);yhi=Math.min(1,Math.ceil((yhi+0.02)*20)/20);
  var y=frameGrid(fig,svg,W,H,L,R,T,B,ylo,yhi,"acc");
  [4096,16384,32768,65536,131072].forEach(function(c){
    el("text",{x:x(c),y:H-8,class:"ticktext","text-anchor":"middle"},svg).textContent=fmtCtx(c);
  });
  cfgs.forEach(function(cfg,ci){
    var ctxs=Object.keys(D.rescue[cfg]).map(Number).sort(function(a,b){return a-b;});
    var pts=ctxs.filter(function(c){return D.rescue[cfg][c].acc!==null;})
      .map(function(c){return [x(c),y(D.rescue[cfg][c].acc)];});
    var st0=D.rescue[cfg][ctxs[0]];
    if(st0&&st0.lo!==null&&st0.hi!==null){
      var top=ctxs.filter(function(c){return D.rescue[cfg][c].acc!==null;})
        .map(function(c){return [x(c),y(D.rescue[cfg][c].hi)];});
      var bot=ctxs.filter(function(c){return D.rescue[cfg][c].acc!==null;})
        .map(function(c){return [x(c),y(D.rescue[cfg][c].lo)];});
      el("path",{d:bandPath(top,bot),class:"band l"+(ci%6)},svg);
    }
    var dash=D.rescue[cfg][ctxs[0]]&&D.rescue[cfg][ctxs[0]].source==="p10";
    el("path",{d:linePath(pts),class:"cl l"+(ci%6),"stroke-dasharray":dash?"6 4":""},svg);
    pts.forEach(function(p){el("circle",{cx:p[0],cy:p[1],r:2.2,class:"l"+(ci%6)},svg);});
    var src=D.rescue[cfg][ctxs[0]]?D.rescue[cfg][ctxs[0]].source:"";
    var sw=document.createElement("span");sw.className="k";
    sw.innerHTML='<span class="sw" style="background:var(--line'+(ci%6)+')"></span>'+esc(cfg)
      +(src?' <span style="color:var(--faint)">('+src+')</span>':"");
    leg.appendChild(sw);
  });
  var cap=document.createElement("div");cap.style.cssText="color:var(--faint);font-size:11px;padding-top:4px";
  cap.textContent="shaded = Wilson 95% CI (p11 rows); dashed line = p10 micro-kv projection. "
    +"Both fixtures are declared synthetic; neither row is a model forward pass (README §3.4).";
  fig.appendChild(cap);
}
/* ================= agent panels (p14) ================= */
function renderAgentBars(fig,leg){
  clearNode(fig);
  var cfgs=Object.keys(D.agent).sort(),ctxs=D.avail.agent;
  if(ctxs.indexOf(S.ctx)<0){
    emptyState(fig,"<b>not measured</b> — the agent harness ran at 32K and 64K only.");
    return;
  }
  var W=560,H=300,L=52,R=10,T=8,B=30;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var max=0.02;
  cfgs.forEach(function(c){max=Math.max(max,D.agent[c][S.ctx].v_schema,D.agent[c][S.ctx].amnesia);});
  max=Math.ceil(max*25)/25;
  var y=frameGrid(fig,svg,W,H,L,R,T,B,0,max,"rate");
  var band=(W-L-R)/cfgs.length;
  cfgs.forEach(function(cfg,i){
    var st=D.agent[cfg][S.ctx],cx=L+band*(i+0.5);
    el("rect",{x:cx-band*0.32,y:y(st.v_schema),width:band*0.28,height:H-B-y(st.v_schema),
      class:"bar l3",rx:2},svg);
    el("rect",{x:cx+band*0.04,y:y(st.amnesia),width:band*0.28,height:Math.max(0.5,H-B-y(st.amnesia)),
      class:"bar l0",rx:2},svg);
    el("text",{x:cx,y:H-16,class:"ticktext","text-anchor":"middle"},svg).textContent=cfg;
    el("text",{x:cx,y:H-4,class:"ticktext","text-anchor":"middle",fill:"#f87171"},svg)
      .textContent=(100*st.v_schema).toFixed(1)+"%";
  });
  var k=document.createElement("span");k.className="k";
  k.innerHTML='<span class="sw" style="background:var(--line3)"></span>V-schema corruption'
    +'&nbsp;&nbsp;<span class="sw" style="background:var(--line0)"></span>K-amnesia (measured null: 0 everywhere)';
  leg.appendChild(k);
  var cap=document.createElement("div");cap.style.cssText="color:var(--faint);font-size:11px;padding-top:4px";
  cap.textContent="K-amnesia is a controlled null: routing flipped in 6/6,400 turn-records, all inside "
    +"the 20K-token forgotten-span window (docs/AGENT_FAILURE_MODES.md).";
  fig.appendChild(cap);
}
function renderAgentResolved(fig,leg){
  clearNode(fig);
  var cfgs=Object.keys(D.agent).sort();
  if(D.avail.agent.indexOf(S.ctx)<0){
    emptyState(fig,"<b>not measured</b> — the agent harness ran at 32K and 64K only.");return;
  }
  var W=560,H=300,L=52,R=10,T=8,B=30;
  var svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%"},fig);
  var y=frameGrid(fig,svg,W,H,L,R,T,B,0,1,"rate");
  var band=(W-L-R)/cfgs.length;
  cfgs.forEach(function(cfg,i){
    var st=D.agent[cfg][S.ctx],cx=L+band*(i+0.5);
    el("rect",{x:cx-band*0.32,y:y(st.resolved),width:band*0.28,height:H-B-y(st.resolved),
      class:"bar l0",rx:2},svg);
    el("rect",{x:cx+band*0.04,y:y(st.collapsed),width:band*0.28,height:H-B-y(st.collapsed),
      class:"bar l1",rx:2},svg);
    el("text",{x:cx,y:H-16,class:"ticktext","text-anchor":"middle"},svg).textContent=cfg;
    el("text",{x:cx,y:H-4,class:"ticktext","text-anchor":"middle"},svg)
      .textContent=(100*st.resolved).toFixed(0)+"% resolved";
  });
  var k=document.createElement("span");k.className="k";
  k.innerHTML='<span class="sw" style="background:var(--line0)"></span>resolved'
    +'&nbsp;&nbsp;<span class="sw" style="background:var(--line1)"></span>loop-collapsed (3-strike abort @ turn 8)';
  leg.appendChild(k);
}
"""
