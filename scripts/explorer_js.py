"""Core JavaScript for docs/explorer.html: state, controls, SVG helpers.

Renderers for the three views live in scripts/explorer_views.py; the builder
concatenates both (function declarations hoist inside one IIFE). No external
libraries: everything is hand-rolled SVG.
"""

JS_CORE = r"""
(function(){
"use strict";
var D = window.KQ_DATA;
function $(id){return document.getElementById(id);}
var S = {ctx:32768, task:"needle", phase:"v1", budget:24};
var CTXS = [[4096,"4K"],[8192,"8K"],[16384,"16K"],[32768,"32K"],[65536,"64K"],[131072,"128K"]];
var TASKS = [["needle","Needle"],["multihop","Multi-Hop"],["longsum","LongSum"],["longcode","LongCode"],["agent","20-Turn Agent"]];
var BUDGETS = [24,48,80];
var MODE_BADGE = {measured:"b-measured","measured (cpu)":"b-measured","replay-baseline":"b-replay",
projected:"b-projected",modeled:"b-modeled","ncu-modeled":"b-modeled",simulated:"b-simulated",
synthetic:"b-projected",analytical:"b-modeled",reference:"b-na"};
function badge(mode,label){
  var cls = MODE_BADGE[String(mode).toLowerCase()] || "b-na";
  return '<span class="badge '+cls+'">'+(label||mode)+'</span>';
}
function el(name,attrs,parent){
  var n = document.createElementNS("http://www.w3.org/2000/svg",name);
  for(var k in attrs){n.setAttribute(k,attrs[k]);}
  if(parent){parent.appendChild(n);}
  return n;
}
function lin(d0,d1,r0,r1){return function(v){return r0+(d1?((v-d0)/(d1-d0)*(r1-r0)):r0);};}
function fmtCtx(c){for(var i=0;i<CTXS.length;i++){if(CTXS[i][0]===c){return CTXS[i][1];}}return (c/1024)+"K";}
function fmtMiB(b){return (b/1048576).toFixed(1);}
function fmtGiB(b){return (b/1073741824).toFixed(2);}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");}
function linePath(pts){
  var d="";for(var i=0;i<pts.length;i++){d+=(i?"L":"M")+pts[i][0].toFixed(1)+" "+pts[i][1].toFixed(1);}
  return d;
}
function bandPath(top,bot){
  if(!top.length||top.length!==bot.length){return "";}
  var d="M"+top[0][0].toFixed(1)+" "+top[0][1].toFixed(1);
  for(var i=1;i<top.length;i++){d+="L"+top[i][0].toFixed(1)+" "+top[i][1].toFixed(1);}
  for(var j=bot.length-1;j>=0;j--){d+="L"+bot[j][0].toFixed(1)+" "+bot[j][1].toFixed(1);}
  return d+"Z";
}
function clearNode(n){while(n.firstChild){n.removeChild(n.firstChild);}}
function emptyState(fig,msg){
  clearNode(fig);
  var d=document.createElement("div");d.className="empty";d.innerHTML=msg;
  fig.appendChild(d);
}
/* ---------- availability ---------- */
function ctxEnabled(ctx){
  if(S.task==="agent"){return D.avail.agent.indexOf(ctx)>=0;}
  var depth=(D.avail.depth[S.task]||[]),frontier=(D.avail.frontier[S.task]||[]);
  if(S.phase==="v2"&&S.task==="needle"){return true;}
  return depth.indexOf(ctx)>=0||frontier.indexOf(ctx)>=0;
}
function noteFor(fig,kind){
  var src={depth:"p4 / p5",frontier:"p6",rescue:"p10 / p11",fleet:"p13",agent:"p14"}[kind];
  return '<span class="src">bench/data — '+src+'</span>';
}
/* ---------- controls ---------- */
function buildControls(){
  var ctxBox=$("ctx-chips"),taskBox=$("task-chips"),budgetBox=$("budget-chips");
  CTXS.forEach(function(c){
    var b=document.createElement("button");b.className="chip";b.textContent=c[1];
    b.id="ctx-"+c[0];b.onclick=function(){S.ctx=c[0];refresh();};ctxBox.appendChild(b);
  });
  TASKS.forEach(function(t){
    var b=document.createElement("button");b.className="chip";b.textContent=t[1];
    b.id="task-"+t[0];b.onclick=function(){S.task=t[0];S.ctx=pickCtx();refresh();};taskBox.appendChild(b);
  });
  BUDGETS.forEach(function(g){
    var b=document.createElement("button");b.className="chip";b.textContent=g+" GB";
    b.id="bud-"+g;b.onclick=function(){S.budget=g;refresh();};budgetBox.appendChild(b);
  });
  $("seg-v1").onclick=function(){S.phase="v1";refresh();};
  $("seg-v2").onclick=function(){S.phase="v2";refresh();};
  $("theme").onclick=function(){
    var root=document.documentElement;
    var cur=root.getAttribute("data-theme")==="light"?"dark":"light";
    root.setAttribute("data-theme",cur);try{localStorage.setItem("kq-theme",cur);}catch(e){}
  };
  try{var saved=localStorage.getItem("kq-theme");if(saved){document.documentElement.setAttribute("data-theme",saved);}}catch(e){}
}
function pickCtx(){
  var avail=CTXS.filter(function(c){return ctxEnabled(c[0]);}).map(function(c){return c[0];});
  if(avail.indexOf(S.ctx)>=0){return S.ctx;}
  return avail.length?avail[Math.max(0,avail.indexOf(65536))>=0?avail.indexOf(65536):avail.length-1]:32768;
}
function syncControls(){
  CTXS.forEach(function(c){
    var b=$("ctx-"+c[0]);if(!b){return;}
    var on=ctxEnabled(c[0]);
    b.disabled=!on;b.classList.toggle("on",on&&S.ctx===c[0]);
    if(on&&S.ctx===c[0]){b.title="";}
  });
  TASKS.forEach(function(t){$("task-"+t[0]).classList.toggle("on",S.task===t[0]);});
  BUDGETS.forEach(function(g){$("bud-"+g).classList.toggle("on",S.budget===g);});
  $("seg-v1").classList.toggle("on",S.phase==="v1");
  $("seg-v2").classList.toggle("on",S.phase==="v2");
}
/* ---------- dispatch ---------- */
function refresh(){
  syncControls();
  renderHeaderSpark();
  renderPanelA();
  renderPanelB();
  renderFleet();
  renderSnippet();
}
function renderHeaderSpark(){
  var host=$("spark");clearNode(host);
  var w=210,h=30,svg=el("svg",{width:w,height:h,viewBox:"0 0 "+w+" "+h},host);
  // p6 calls its FP16 baseline k16v16_post_rope_s4 (not "fp16").
  var fp=D.frontier.needle["k16v16_post_rope_s4"]||{},k44=D.frontier.needle["k4v4_post_rope_s4"]||{};
  var ctxs=[4096,16384,32768,65536,131072].filter(function(c){return fp[c];});
  if(!ctxs.length){return;}
  var x=lin(ctxs[0],ctxs[ctxs.length-1],4,w-8),y=lin(0.70,1.0,h-6,6);
  function draw(key,cls){
    var pts=ctxs.map(function(c){return [x(c),y(fp[c].acc)];});
    el("path",{d:linePath(pts),class:"cl "+cls,fill:"none",strokeWidth:2},svg);
  }
  draw(0,"l0");el("circle",{cx:x(ctxs[ctxs.length-1]),cy:y(fp[ctxs[ctxs.length-1]].acc),r:2.4,class:"l0"},svg);
  var pts2=ctxs.map(function(c){return k44[c]?k44[c].acc:null;}).filter(function(v){return v!==null;});
  if(pts2.length===ctxs.length){
    var d=ctxs.map(function(c){return [x(c),y(k44[c].acc)];});
    el("path",{d:linePath(d),class:"cl l3",fill:"none",strokeWidth:2},svg);
  }
  el("text",{x:4,y:h-1,class:"ticktext"},svg).textContent="FP16";
  el("text",{x:w-46,y:h-1,class:"ticktext",fill:"#f87171"},svg).textContent="INT4-K";
}
"""

# Concatenated AFTER the view modules so they land inside the IIFE.
JS_BOOT = r"""
buildControls();
refresh();
}).call(this);
"""
