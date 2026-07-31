#!/usr/bin/env node
/*
 * Screenshot a page that draws with WebGL.
 *
 * `chrome --headless --screenshot` fires on a virtual-time deadline, which
 * expires while the CDN modules and the .glb files are still in flight, and
 * lands on an empty canvas every time. This drives the same headless Chrome
 * over DevTools instead and waits for the page to say it is done — set
 * document.title to "ready" — before asking for pixels.
 *
 *   node tools/shoot.mjs <url> <out.png> [width] [height]
 */
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, out, w = "1400", h = "900"] = process.argv.slice(2);
if (!url || !out) {
  console.error("usage: node tools/shoot.mjs <url> <out.png> [width] [height]");
  process.exit(1);
}

const PORT = 9222 + (process.pid % 500);
const profile = mkdtempSync(join(tmpdir(), "shoot-"));

const chrome = spawn("google-chrome-stable", [
  /* software GL, spelled the way current Chrome wants it: --disable-gpu kills
     the context entirely, and plain --use-gl=swiftshader no longer binds */
  "--headless=new", "--use-gl=angle", "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader", "--disable-dev-shm-usage",
  "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${w},${h}`, "about:blank"
], { stdio: ["ignore", "ignore", "pipe"] });

const bail = (msg) => { chrome.kill(); console.error(msg); process.exit(1); };
setTimeout(() => bail("timed out"), 90000).unref();

/* the debugging port takes a moment to open */
let target;
for (let i = 0; i < 60; i++) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    const list = await r.json();
    target = list.find((t) => t.type === "page");
    if (target) break;
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 250));
}
if (!target) bail("chrome never opened a debugging port");

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

let id = 0;
const waiting = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m.result); waiting.delete(m.id); }
  /* the page's own diagnostics, so a silent failure is not a blank png */
  if (m.method === "Runtime.consoleAPICalled") {
    const a = (m.params.args || []).map((x) => x.value ?? x.description ?? x.type);
    console.error(`  [${m.params.type}]`, a.join(" "));
  }
  if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    console.error("  [error]", d.exception?.description || d.text);
  }
  if (m.method === "Log.entryAdded") {
    console.error(`  [${m.params.entry.level}]`, m.params.entry.text);
  }
};
const send = (method, params = {}) => new Promise((res) => {
  const n = ++id;
  waiting.set(n, res);
  ws.send(JSON.stringify({ id: n, method, params }));
});

await send("Page.enable");
await send("Runtime.enable");
await send("Log.enable");
await send("Emulation.setDeviceMetricsOverride",
  { width: +w, height: +h, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });

/* poll for the page's own done signal rather than guessing a delay */
const deadline = Date.now() + 60000;
let ready = false;
while (Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 300));
  const r = await send("Runtime.evaluate",
    { expression: "document.title", returnByValue: true });
  if (r?.result?.value === "ready") { ready = true; break; }
}
if (!ready) console.error("warning: page never reported ready — shooting anyway");

/* one more frame, so the last render is composited */
await send("Runtime.evaluate", {
  expression: "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))",
  awaitPromise: true
});

const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.data, "base64"));
console.log(`${out}  ${(Buffer.from(shot.data, "base64").length / 1024) | 0} KB`);

ws.close();
chrome.kill();
