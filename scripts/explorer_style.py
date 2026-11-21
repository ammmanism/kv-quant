"""Inline CSS for docs/explorer.html (no external resources).

Dark instrument-panel theme with a light/print toggle. Kept as a Python
module so the whole explorer stays a single generated artifact with zero
dependencies; the drift gate rebuilds it byte-for-byte.
"""

CSS = r"""
:root{--bg:#0a0e14;--panel:#10161f;--panel2:#151d29;--ink:#d9e2ec;--muted:#8494a7;
--faint:#5b6b7d;--teal:#2dd4bf;--amber:#fbbf24;--red:#f87171;--blue:#60a5fa;
--violet:#c084fc;--grid:#1d2836;--line0:#2dd4bf;--line1:#fbbf24;--line2:#60a5fa;
--line3:#f87171;--line4:#c084fc;--line5:#a3e635}
[data-theme="light"]{--bg:#f6f8fa;--panel:#ffffff;--panel2:#eef2f6;--ink:#1c2733;
--muted:#54657a;--faint:#8195aa;--grid:#dfe6ee}
*{box-sizing:border-box;margin:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,
-apple-system,sans-serif;min-height:100vh}
a{color:var(--teal)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header{position:sticky;top:0;z-index:9;background:linear-gradient(180deg,
var(--panel) 0%,var(--panel2) 100%);border-bottom:1px solid var(--grid);
padding:14px 0 10px;backdrop-filter:blur(4px)}
.brand{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px}
.brand h1{font-size:19px;letter-spacing:.4px;font-weight:650}
.brand h1 .amp{color:var(--teal)}
.prov{color:var(--muted);font-size:12.5px;display:flex;flex-wrap:wrap;gap:4px 14px;
margin-top:3px;align-items:center}
.prov b{color:var(--ink);font-weight:600}
.sha{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--faint)}
.spark{display:inline-flex;align-items:center;gap:8px}
.spark svg{vertical-align:middle}
/* ---- controls ---- */
.controls{display:flex;flex-wrap:wrap;gap:14px 26px;padding:12px 0 4px}
.ctl{display:flex;align-items:center;gap:8px}
.ctl label{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
color:var(--faint);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{background:transparent;border:1px solid var(--grid);color:var(--muted);
padding:4px 10px;border-radius:6px;font-size:12.5px;cursor:pointer;
transition:border-color .15s,color .15s,background .15s;font-family:inherit}
.chip:hover:not(:disabled){border-color:var(--teal);color:var(--ink)}
.chip.on{background:var(--teal);border-color:var(--teal);color:#04211d;font-weight:600}
.chip:disabled{opacity:.28;cursor:not-allowed;text-decoration:line-through}
.seg{display:flex;border:1px solid var(--grid);border-radius:7px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--muted);padding:5px 12px;
font-size:12.5px;cursor:pointer;font-family:inherit;transition:background .15s}
.seg button.on{background:var(--amber);color:#241a02;font-weight:650}
.seg button:not(.on):hover{color:var(--ink)}
/* ---- views ---- */
main{padding:8px 0 30px}
.view{background:var(--panel);border:1px solid var(--grid);border-radius:10px;
padding:18px 20px;margin:16px 0;box-shadow:0 1px 0 rgba(0,0,0,.25)}
.view>h2{font-size:15px;letter-spacing:.3px;margin-bottom:2px}
.view>h2 .no{color:var(--teal);font-family:ui-monospace,Consolas,monospace}
.view>.sub{color:var(--muted);font-size:12.5px;margin-bottom:12px}
.panelgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:960px){.panelgrid{grid-template-columns:1fr}}
.panel{background:var(--panel2);border:1px solid var(--grid);border-radius:8px;
padding:12px 12px 6px;min-width:0}
.panel h3{font-size:12px;color:var(--muted);text-transform:uppercase;
letter-spacing:1px;font-weight:600;margin-bottom:8px}
.panel h3 .src{float:right;text-transform:none;letter-spacing:0}
figure{margin:0}
figcaption{color:var(--faint);font-size:11.5px;padding:6px 2px 8px}
svg text{font-family:ui-monospace,Consolas,monospace}
svg .axis{stroke:var(--faint);stroke-width:1}
svg .gridline{stroke:var(--grid);stroke-width:1}
svg .ticktext{fill:var(--muted);font-size:10px}
svg .ylab,.xlab{fill:var(--faint);font-size:10px}
svg .l0{stroke:var(--line0)}svg .l1{stroke:var(--line1)}svg .l2{stroke:var(--line2)}
svg .l3{stroke:var(--line3)}svg .l4{stroke:var(--line4)}svg .l5{stroke:var(--line5)}
svg .cl{fill:none;stroke-width:2;stroke-linejoin:round}
svg .band{opacity:.13}
svg .bar{opacity:.9}
svg .cross{stroke:var(--ink);stroke-width:.8;stroke-dasharray:3 3;opacity:.6}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:11.5px;
color:var(--muted);padding:2px 2px 8px}
.legend .k{display:inline-flex;align-items:center;gap:5px}
.legend .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.empty{color:var(--faint);font-size:12px;padding:26px 10px;text-align:center;
border:1px dashed var(--grid);border-radius:6px}
.empty b{color:var(--muted)}
/* ---- badges ---- */
.badge{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;
border-radius:20px;padding:1.5px 9px;font-weight:600;letter-spacing:.4px;
text-transform:uppercase;border:1px solid}
.badge::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.b-measured{color:#34d399;border-color:#34d39955;background:#34d39911}
.b-projected{color:#fbbf24;border-color:#fbbf2455;background:#fbbf2411}
.b-replay{color:#c084fc;border-color:#c084fc55;background:#c084fc11}
.b-modeled{color:#60a5fa;border-color:#60a5fa55;background:#60a5fa11}
.b-simulated{color:#f472b6;border-color:#f472b655;background:#f472b611}
.b-na{color:var(--faint);border-color:var(--grid);background:transparent}
.badgerow{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}
/* ---- fleet view ---- */
.fleetgrid{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}
@media(max-width:960px){.fleetgrid{grid-template-columns:1fr}}
.stat{display:flex;flex-direction:column;gap:1px;padding:8px 12px;
background:var(--panel2);border:1px solid var(--grid);border-radius:8px}
.stat .v{font-size:22px;font-weight:700;font-family:ui-monospace,Consolas,monospace}
.stat .k{font-size:10.5px;color:var(--faint);text-transform:uppercase;letter-spacing:1px}
.stat .d{font-size:11px;color:var(--muted)}
.statrow{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
@media(max-width:720px){.statrow{grid-template-columns:1fr}}
/* ---- snippet view ---- */
.snipwrap{position:relative}
pre.snip{background:#05080d;border:1px solid var(--grid);border-radius:8px;
padding:14px 16px;overflow-x:auto;font:12.5px/1.55 ui-monospace,Consolas,monospace;
color:#cfe3ee;margin:0}
[data-theme="light"] pre.snip{background:#0d1420;color:#cfe3ee}
pre.snip .c{color:#5b6b7d}
pre.snip .s{color:#a3e635}
pre.snip .n{color:#fbbf24}
.copy{position:absolute;top:8px;right:8px;background:var(--panel2);
border:1px solid var(--grid);color:var(--muted);border-radius:6px;padding:4px 10px;
font-size:11.5px;cursor:pointer;font-family:inherit}
.copy:hover{color:var(--ink);border-color:var(--teal)}
.warn{color:var(--amber);font-size:12px;border-left:3px solid var(--amber);
padding:6px 10px;background:#fbbf240d;border-radius:0 6px 6px 0;margin-top:10px}
.ok{color:#34d399;font-size:12px;margin-top:8px}
/* ---- footer ---- */
footer{border-top:1px solid var(--grid);margin-top:10px;padding:18px 0 40px;
color:var(--muted);font-size:12.5px}
footer h2{font-size:13px;color:var(--ink);margin-bottom:8px;letter-spacing:.4px}
table.provtab{border-collapse:collapse;width:100%;font-size:11.5px}
.provtab th,.provtab td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}
.provtab th{color:var(--faint);text-transform:uppercase;letter-spacing:1px;font-size:10px}
.provtab td.mono{font-family:ui-monospace,Consolas,monospace;color:var(--faint)}
.modebar{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px}
.footnote{color:var(--faint);font-size:11.5px;margin-top:10px}
.theme-toggle{position:absolute;top:14px;right:20px;background:transparent;
border:1px solid var(--grid);color:var(--muted);border-radius:6px;padding:3px 10px;
font-size:11.5px;cursor:pointer;font-family:inherit}
.theme-toggle:hover{color:var(--ink);border-color:var(--teal)}
@media print{header{position:static}.view{break-inside:avoid;box-shadow:none}
.theme-toggle,.copy{display:none}body{background:#fff}}
"""
