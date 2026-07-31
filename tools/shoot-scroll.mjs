#!/usr/bin/env node
/*
 * Drive index.html down the page and photograph the bay at each stop, so the
 * print sequence can be checked frame by frame without a human scrolling.
 *
 *   node tools/shoot-scroll.mjs <url> <outdir> [fractions...]
 *
 * Each fraction is a share of the scrollable height. The page is given time to
 * let the scrub catch up before the shutter, because the whole point of the
 * scrub is that it lags the scrollbar.
 */
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, outdir, ...rest] = process.argv.slice(2);
if (!url || !outdir) {
  console.error("usage: node tools/shoot-scroll.mjs <url> <outdir> [fractions...]");
  process.exit(1);
}
const stops = rest.length ? rest.map(Number) : [0, 0.08, 0.16, 0.24, 0.32];
mkdirSync(outdir, { recursive: true });

const PORT = 9222 + (process.pid % 500);
const profile = mkdtempSync(join(tmpdir(), "shoot-"));
const W = 1440, H = 900;

const chrome = spawn("google-chrome-stable", [
  "--headless=new", "--use-gl=angle", "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader", "--disable-dev-shm-usage",
  "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${W},${H}`, "about:blank"
], { stdio: ["ignore", "ignore", "ignore"] });

const bail = (m) => { chrome.kill(); console.error(m); process.exit(1); };
setTimeout(() => bail("timed out"), 180000).unref();

let target;
for (let i = 0; i < 60; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    target = list.find((t) => t.type === "page");
    if (target) break;
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 250));
}
if (!target) bail("no debugging port");

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

let id = 0;
const waiting = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m.result); waiting.delete(m.id); }
  if (m.method === "Runtime.exceptionThrown")
    console.error("  [error]", m.params.exceptionDetails.exception?.description);
};
const send = (method, params = {}) => new Promise((res) => {
  const n = ++id; waiting.set(n, res);
  ws.send(JSON.stringify({ id: n, method, params }));
});
const evalIn = async (expression, awaitPromise = false) =>
  (await send("Runtime.evaluate", { expression, awaitPromise, returnByValue: true }))
    ?.result?.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride",
  { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });

/* wait for the machine to have loaded and drawn at least once */
for (let i = 0; i < 120; i++) {
  await new Promise((r) => setTimeout(r, 500));
  const ok = await evalIn(
    `!!document.querySelector('#bayCanvas') &&
     document.getElementById('bayStateT').textContent`);
  if (ok && ok !== "Offline") break;
}

for (const f of stops) {
  await evalIn(`window.scrollTo(0, document.body.scrollHeight * ${f})`);
  /* the scrub has a 0.7 s time constant; give it several of those */
  await new Promise((r) => setTimeout(r, 3500));
  const state = await evalIn(
    `[document.getElementById('bayStateT').textContent,
      document.getElementById('telPct').textContent,
      document.getElementById('telLayer').textContent].join(' | ')`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  const name = join(outdir, `s${String(f).replace(".", "_")}.png`);
  writeFileSync(name, Buffer.from(shot.data, "base64"));
  console.log(`${f.toFixed(2)}  ${state}`);
}

ws.close();
chrome.kill();
