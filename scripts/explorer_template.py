"""HTML skeleton for docs/explorer.html (single file, zero dependencies).

The provenance footer is rendered server-side from the verified payload so
the table is readable without JavaScript; all charts are client-side SVG.
"""

from __future__ import annotations

from explorer_js import JS_BOOT, JS_CORE
from explorer_style import CSS
from explorer_views import JS_VIEWS
from explorer_views2 import JS_VIEWS2

_PROV_ROW = (
    '<tr><td class="mono">{file}</td><td>{rows:,}</td><td>{mode}</td>'
    '<td class="mono">{model}</td><td class="mono">{sha12}…</td></tr>'
)


def _prov_table(artifacts: list[dict[str, object]]) -> str:
    rows = "\n".join(
        _PROV_ROW.format(
            file=a["file"],
            rows=int(a["rows"]),  # type: ignore[arg-type]
            mode=a["mode"],  # type: ignore[arg-type]
            model=(a["model_id"] or "—"),  # type: ignore[index,operator]
            sha12=str(a["sha256"])[:12],  # type: ignore[index]
        )
        for a in artifacts
    )
    return (
        '<table class="provtab"><tr><th>artifact</th><th>rows</th>'
        f"<th>epistemic mode</th><th>model / fixture</th><th>sha256</th></tr>\n{rows}</table>"
    )


def render_html(payload_json: str, artifacts: list[dict[str, object]]) -> str:
    """Assemble the full self-contained document from payload + modules."""
    table = _prov_table(artifacts)
    app_js = JS_CORE + JS_VIEWS + JS_VIEWS2 + JS_BOOT
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kv-quant v2.0.0 — interactive KV-quantization explorer</title>
<meta name="description" content="Zero-dependency interactive explorer over the
SHA256-verified kv-quant benchmark CSVs: the late-context INT4-K cliff, the
v2.0 Hadamard rescue, agent drift, and fleet concurrency.">
<style>{CSS}</style>
</head>
<body>
<header><div class="wrap">
<button class="theme-toggle" id="theme" type="button">◐ theme</button>
<div class="brand"><h1>kv<span class="amp">-</span>quant <span style="color:var(--muted);font-weight:400">explorer</span></h1>
<span class="badge b-measured">v2.0.0</span></div>
<div class="prov">
<b>Qwen/Qwen2.5-7B-Instruct</b> · 7.61 B params · GQA <b>7:1</b> (28 Q / 4 KV heads) · rotary_dim <b>128</b>
<span class="spark" id="spark"></span>
</div>
<div class="prov" style="margin-top:2px">
<span><b>{len(artifacts)}</b> artifacts · <b>{len(artifacts)}</b> SHA256-verified against manifests ·
latency/ITL rows modeled for <b>meta-llama/Llama-2-7b-hf</b> (p7) ·
rebuild gate: <span class="sha">scripts/check_explorer_drift.py</span></span>
</div>
<div class="controls">
<div class="ctl"><label>context</label><div class="chips" id="ctx-chips"></div></div>
<div class="ctl"><label>task</label><div class="chips" id="task-chips"></div></div>
<div class="ctl"><label>phase</label><div class="seg">
<button type="button" id="seg-v1">v1.0 Baseline</button>
<button type="button" id="seg-v2">v2.0 Hadamard + Head-Adaptive</button>
</div></div>
<div class="ctl"><label>vram budget</label><div class="chips" id="budget-chips"></div></div>
</div>
</div></header>
<main class="wrap">
<section class="view">
<h2><span class="no">①</span> Position-Resolved Cliff &amp; ΔlogP Explorer</h2>
<div class="sub">Where in the context the quantization cliff bites, per configuration.
Greyed contexts are honest absences — nothing is interpolated.</div>
<div class="panelgrid">
<div class="panel"><h3 id="h3-a"></h3><div class="badgerow" id="badges-a"></div>
<div class="legend" id="leg-a"></div><figure id="fig-a"></figure></div>
<div class="panel"><h3 id="h3-b"></h3><div class="badgerow" id="badges-b"></div>
<div class="legend" id="leg-b"></div><figure id="fig-b"></figure></div>
</div>
</section>
<section class="view">
<h2><span class="no">②</span> Ragged-Batch Concurrent Agent Session &amp; Fleet Cost Calculator</h2>
<div class="sub">How many simultaneous 20-turn sessions fit a GPU budget with
copy-on-write paged prefix sharing (p13, modeled byte model).</div>
<div class="badgerow" id="badges-c"></div>
<div class="statrow" id="fleet-stats"></div>
<div class="fleetgrid">
<div class="panel"><h3 id="h3-c"></h3><div class="legend" id="leg-c"></div><figure id="fig-c"></figure></div>
</div>
</section>
<section class="view">
<h2><span class="no">③</span> One-Click QuantConfig Snippet</h2>
<div class="sub">A cited, compile-checked configuration for the current selection —
with the AVOID patterns the CSVs actually measured.</div>
<div class="badgerow" id="badges-d"></div>
<div class="snipwrap">
<button class="copy" id="copy" type="button">copy</button>
<pre class="snip" id="snip"></pre>
</div>
<div class="warn" id="snip-warn" style="display:none"></div>
<div class="ok" id="snip-ok"></div>
</section>
</main>
<footer><div class="wrap">
<h2>Evidence provenance — every number resolves to a committed, SHA256-verified CSV</h2>
<div class="modebar">
<span class="badge b-measured">measured (CPU objects)</span>
<span class="badge b-replay">replay-baseline (modeled quality)</span>
<span class="badge b-projected">projected (synthetic fixture)</span>
<span class="badge b-modeled">modeled (analytical)</span>
<span class="badge b-simulated">simulated harness</span>
</div>
{table}
<p class="footnote">Modeled rows: p7 latency/ITL use analytical byte equations for
Llama-2-7b-hf; p13 fleet sessions use the paged-cache byte model for the Qwen geometry.
Projected rows: p10/p11 v2.0 rescue curves come from declared synthetic fixtures via a
replay adapter — not model forward passes. Simulated rows: p14 drives a 20-turn ReAct
harness with measured quantizer error; no LLM ran on the host (ADR-012).
This file is regenerated and byte-compared by <code>make verify-explorer</code>;
it can never silently drift from <code>bench/data/*.csv</code>.</p>
</div></footer>
<script>window.KQ_DATA = {payload_json};</script>
<script>{app_js}</script>
</body>
</html>
"""
