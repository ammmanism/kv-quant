/* Headless smoke driver for docs/explorer.html.
 *
 * Stubs a minimal DOM, boots the embedded app, then drives every
 * task x phase x context x budget combination via the stored onclick
 * handlers. Exits non-zero on any runtime error or empty figure.
 *
 * Used by tests/test_explorer.py (skipped when node is unavailable);
 * also runnable directly:  node scripts/explorer_smoke.js
 */
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "docs", "explorer.html"), "utf8");

// --- extract payload ---
const marker = "window.KQ_DATA = ";
const s = html.indexOf(marker);
const raw = html.slice(s + marker.length, html.indexOf("</script>", s)).replace(/;\s*$/, "");
const data = JSON.parse(raw);

// --- minimal DOM stubs ---
function makeEl() {
  return {
    children: [],
    style: { cssText: "", setProperty() {} },
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    appendChild(c) { this.children.push(c); return c; },
    removeChild() { this.children.shift(); },
    get firstChild() { return this.children[0] || null; },
    setAttribute() {}, getAttribute() { return null; },
    addEventListener() {},
    textContent: "", innerHTML: "",
    onclick: null, disabled: false, title: "", id: "",
  };
}
const cache = {};
const registry = [];
global.window = { KQ_DATA: data };
global.document = {
  getElementById(id) {
    if (cache[id]) return cache[id];
    const found = registry.find((e) => e.id === id);
    if (found) { cache[id] = found; return found; }
    cache[id] = makeEl();
    registry.push(cache[id]);
    return cache[id];
  },
  createElement() { const e = makeEl(); registry.push(e); return e; },
  createElementNS() { const e = makeEl(); registry.push(e); return e; },
  createTextNode(t) { return { textContent: t }; },
  body: makeEl(),
  documentElement: makeEl(),
};
global.navigator = { clipboard: { writeText: () => Promise.resolve() } };

// --- boot the app ---
const jsStart = html.lastIndexOf("<script>");
const js = html.slice(jsStart + 8, html.indexOf("</script>", jsStart));
try {
  new Function(js).call(global.window);
} catch (e) {
  console.error("BOOT FAILED:", e.stack);
  process.exit(1);
}
console.log("boot OK; default render done");
if (cache["spark"].children.length === 0) {
  console.error("HEADER SPARK BLANK — FP16 baseline missing from D.frontier.needle");
  process.exit(1);
}
if (cache["snip-warn"] && cache["snip-warn"].style.display === "none") {
  // v1/32K has no warning by design; checked below on the 64K AVOID branch.
}

// --- drive every combo ---
const tasks = ["needle", "multihop", "longsum", "longcode", "agent"];
const ctxs = [4096, 8192, 16384, 32768, 65536, 131072];
let calls = 0, skippedCtx = 0;
for (const task of tasks) {
  cache["task-" + task].onclick();
  for (const phase of ["v1", "v2"]) {
    cache["seg-" + phase].onclick();
    for (const ctx of ctxs) {
      const btn = cache["ctx-" + ctx];
      if (!btn || btn.disabled) { skippedCtx++; continue; }
      btn.onclick();
      for (const bud of [24, 48, 80]) {
        cache["bud-" + bud].onclick();
        calls++;
        for (const fig of ["fig-a", "fig-b", "fig-c"]) {
          if (cache[fig].children.length === 0) {
            console.error(`EMPTY FIGURE ${fig} @ ${task}/${phase}/${ctx}/${bud}`);
            process.exit(1);
          }
        }
        if (!cache["snip"].textContent.includes("QuantConfig(")) {
          console.error(`BAD SNIPPET @ ${task}/${phase}/${ctx}/${bud}`);
          process.exit(1);
        }
        const wantWarn = task === "agent" || phase === "v2" || ctx >= 65536;
        if (cache["snip-warn"].style.display !== (wantWarn ? "block" : "none")) {
          console.error(`HIDDEN/SHOWN WARNING MISMATCH @ ${task}/${phase}/${ctx}/${bud}`);
          process.exit(1);
        }
      }
    }
  }
}
console.log(`drove ${calls} combos, ${skippedCtx} honestly-disabled ctx chips — no runtime errors`);

// --- canonical snippets for the Python-side compile checks ---
var cases = [
  ["v1", "needle", 32768, 24],
  ["v2-density", "needle", 32768, 24],
  ["v2-quality", "needle", 65536, 48],
  ["agent", "agent", 32768, 24],
];
for (const [name, task, ctx, bud] of cases) {
  cache["task-" + task].onclick();
  cache["seg-" + (name.indexOf("v2") === 0 ? "v2" : "v1")].onclick();
  cache["ctx-" + ctx].onclick();
  cache["bud-" + bud].onclick();
  if (cache["snip"].textContent.indexOf("QuantConfig(") < 0) {
    console.error(`BAD SNIPPET for ${name}`);
    process.exit(1);
  }
  console.log("SNIPPET-BEGIN " + name);
  console.log(cache["snip"].textContent);
  console.log("SNIPPET-END " + name);
}
console.log("SMOKE OK");
