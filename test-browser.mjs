// Browser behaviour checks. Runs the real page in a real (headless) Chromium
// and drives the widgets: clicks, keyboard, scroll, timers, computed styles.
//
// No dependencies. Node 22 gives us a global WebSocket and fetch; we speak the
// Chrome DevTools Protocol directly over the WebSocket Chrome prints on
// startup. The browser is found in the usual places, or via $CHROME_PATH.
//
//   npm run test:browser
//
// If no browser is installed the run prints SKIP and exits 0 — the static and
// math checks in test-guide.mjs stay the hard gate, and a machine without a
// browser should not be red for a reason it cannot fix.

import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, normalize, resolve, sep } from "node:path";

const ROOT = resolve(new URL(".", import.meta.url).pathname);
const VIEWPORT = { width: 1280, height: 900 };
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------- browser ---

function findBrowser() {
  const candidates = [process.env.CHROME_PATH];
  const cache = join(process.env.HOME ?? "", "Library/Caches/ms-playwright");
  if (existsSync(cache)) {
    for (const dir of readdirSync(cache).filter(d => d.startsWith("chromium-")).sort().reverse()) {
      for (const app of readdirSync(join(cache, dir)).filter(d => d.endsWith(".app"))) {
        candidates.push(join(cache, dir, app, "Contents/MacOS", app.replace(/\.app$/, "")));
      }
    }
  }
  candidates.push(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
  );
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    try {
      candidates.push(execFileSync("which", [name], { encoding: "utf8" }).trim());
    } catch { /* not on PATH */ }
  }
  return candidates.find(path => path && existsSync(path)) ?? null;
}

// ----------------------------------------------------------------- server ---
// ES modules will not load over file://, so the guide needs an origin. This is
// the same static serve as `npm run serve`, in-process and on a free port.

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
};

function startServer() {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
    const file = join(ROOT, path === "/" ? "index.html" : path);
    if (!file.startsWith(ROOT + sep) || !existsSync(file)) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  });
  return new Promise(done => server.listen(0, "127.0.0.1", () =>
    done({ url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() })));
}

// -------------------------------------------------------------------- CDP ---

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    const events = [];
    const waiters = [];
    let nextId = 0;
    ws.addEventListener("error", reject);
    ws.addEventListener("message", ({ data }) => {
      const msg = JSON.parse(data);
      if (msg.id !== undefined && pending.has(msg.id)) {
        const { resolve: ok, reject: no } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? no(new Error(msg.error.message)) : ok(msg.result);
        return;
      }
      if (!msg.method) return;
      events.push(msg);
      for (let i = waiters.length - 1; i >= 0; i--) {
        if (waiters[i].method !== msg.method) continue;
        clearTimeout(waiters[i].timer);
        waiters[i].resolve(msg.params);
        waiters.splice(i, 1);
      }
    });
    ws.addEventListener("open", () => resolve({
      events,
      close: () => ws.close(),
      send(method, params = {}, sessionId) {
        const id = ++nextId;
        return new Promise((ok, no) => {
          pending.set(id, { resolve: ok, reject: no });
          ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
        });
      },
      waitFor(method, timeout = 20000) {
        const seen = events.find(e => e.method === method);
        if (seen) return Promise.resolve(seen.params);
        return new Promise((ok, no) => {
          const waiter = { method, resolve: ok };
          waiter.timer = setTimeout(() => {
            waiters.splice(waiters.indexOf(waiter), 1);
            no(new Error(`timed out waiting for ${method}`));
          }, timeout);
          waiters.push(waiter);
        });
      },
    }));
  });
}

const KEYS = {
  ArrowRight: { code: "ArrowRight", vk: 39 },
  ArrowLeft: { code: "ArrowLeft", vk: 37 },
  " ": { code: "Space", vk: 32, text: " " },
  r: { code: "KeyR", vk: 82, text: "r" },
};

async function launch(browser) {
  const profile = mkdtempSync(join(tmpdir(), "fa2-cdp-"));
  const proc = spawn(browser, [
    "--headless=new",
    "--remote-debugging-port=0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-gpu",
    `--user-data-dir=${profile}`,
    "about:blank",
  ], { stdio: ["ignore", "pipe", "pipe"] });

  const wsUrl = await new Promise((ok, no) => {
    let buf = "";
    const timer = setTimeout(() => no(new Error("browser never printed a devtools URL")), 30000);
    proc.stderr.on("data", chunk => {
      buf += chunk;
      const match = buf.match(/ws:\/\/\S+/);
      if (!match) return;
      clearTimeout(timer);
      ok(match[0]);
    });
    proc.on("exit", code => {
      clearTimeout(timer);
      no(new Error(`browser exited with code ${code}`));
    });
  });

  const cdp = await connect(wsUrl);
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const send = (method, params) => cdp.send(method, params, sessionId);

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Network.enable");
  await send("Emulation.setDeviceMetricsOverride", { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setFocusEmulationEnabled", { enabled: true });

  const q = sel => JSON.stringify(sel);

  const page = {
    events: cdp.events,
    send,
    async eval(expression) {
      const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
      if (result.exceptionDetails) {
        const { exception, text } = result.exceptionDetails;
        throw new Error(`${exception?.description ?? text} — while running: ${expression.slice(0, 90)}`);
      }
      return result.result.value;
    },
    /** textContent, or null when the node is missing. */
    text: sel => page.eval(`(document.querySelector(${q(sel)}) || {}).textContent ?? null`),
    exists: sel => page.eval(`!!document.querySelector(${q(sel)})`),
    count: sel => page.eval(`document.querySelectorAll(${q(sel)}).length`),
    click: sel => page.eval(`document.querySelector(${q(sel)}).click()`),
    focus: sel => page.eval(`document.querySelector(${q(sel)}).focus()`),
    /** Set a range input's value and fire the `input` event the widgets listen for. */
    setRange: (sel, value) => page.eval(`(() => {
      const el = document.querySelector(${q(sel)});
      el.value = ${JSON.stringify(String(value))};
      el.dispatchEvent(new Event("input", { bubbles: true }));
    })()`),
    async press(key) {
      const spec = KEYS[key];
      if (!spec) throw new Error(`no key spec for ${key}`);
      const base = { key, code: spec.code, windowsVirtualKeyCode: spec.vk, nativeVirtualKeyCode: spec.vk };
      await send("Input.dispatchKeyEvent", { type: spec.text ? "keyDown" : "rawKeyDown", ...base, text: spec.text });
      await send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    },
    /** Scroll a section into the band the progress rail's IntersectionObserver watches. */
    async scrollToSection(id) {
      await page.eval(`(() => {
        document.documentElement.style.scrollBehavior = "auto";
        const top = document.getElementById(${q(id)}).getBoundingClientRect().top + window.scrollY;
        window.scrollTo(0, Math.round(top - window.innerHeight * 0.25));
      })()`);
      await sleep(300);
    },
    async setReducedMotion(reduce) {
      await send("Emulation.setEmulatedMedia", {
        features: [{ name: "prefers-reduced-motion", value: reduce ? "reduce" : "no-preference" }],
      });
    },
    async goto(url) {
      const loaded = cdp.waitFor("Page.loadEventFired");
      await send("Page.navigate", { url });
      await loaded;
      await sleep(150);
    },
    async close() {
      cdp.close();
      proc.kill("SIGKILL");
      await new Promise(done => (proc.exitCode === null ? proc.once("exit", done) : done()));
      rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    },
  };
  return page;
}

// ------------------------------------------------------------------ runner ---

let failures = 0;
let ran = 0;

async function check(name, fn) {
  ran++;
  try {
    await fn();
    console.log(`  ok    ${name}`);
  } catch (error) {
    failures++;
    console.log(`  FAIL  ${name}\n          ${error.message.split("\n").slice(0, 6).join("\n          ")}`);
  }
}

const section = title => console.log(`\n${title}`);

// ------------------------------------------------------------------- main ---

const browser = findBrowser();
if (!browser) {
  console.log("SKIP: no Chrome or Chromium found. Set $CHROME_PATH and re-run.");
  process.exit(0);
}

const server = await startServer();
const page = await launch(browser);

try {
  await page.goto(server.url);
  console.log(`browser checks in ${browser.replace(process.env.HOME ?? "", "~")}`);

  // ------------------------------------------------------------- wiring ---
  section("page health");

  await check("no uncaught exceptions, console errors, or log errors", async () => {
    const thrown = page.events
      .filter(e => e.method === "Runtime.exceptionThrown")
      .map(e => e.params.exceptionDetails.exception?.description ?? e.params.exceptionDetails.text);
    const logged = page.events
      .filter(e => e.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(e.params.type))
      .map(e => e.params.args.map(a => a.value ?? a.description).join(" "));
    const severe = page.events
      .filter(e => e.method === "Log.entryAdded" && e.params.entry.level === "error")
      .map(e => e.params.entry.text);
    const problems = [...thrown, ...logged, ...severe];
    assert.deepEqual(problems, [], `page logged ${problems.length} problem(s):\n` + problems.join("\n"));
  });

  await check("every request stays on the local origin", async () => {
    const external = page.events
      .filter(e => e.method === "Network.requestWillBeSent")
      .map(e => e.params.request.url)
      .filter(url => !url.startsWith(server.url));
    assert.deepEqual(external, [], "the guide must not reach off-origin");
  });

  await check("every response is a 2xx", async () => {
    const bad = page.events
      .filter(e => e.method === "Network.responseReceived")
      .map(e => `${e.params.response.status} ${e.params.response.url}`)
      .filter(line => !line.startsWith("2"));
    assert.deepEqual(bad, [], "the guide should request only files that exist");
  });

  await check("the ES module ran, not merely parsed", async () =>
    assert.equal(await page.count("#work-diagram .work-cell"), 4));

  // ------------------------------------------- chapter 1: attention calc ---
  section("chapter 1 · attention calculator");

  await check("opens on step 1 with prev disabled", async () => {
    assert.equal(await page.text("#att-step-indicator"), "step 1 / 3");
    assert.equal(await page.eval('document.querySelector("#att-prev").disabled'), true);
    assert.equal(await page.eval('document.querySelector("#att-next").disabled'), false);
  });

  await check("the demo reproduces the numbers printed in the text", async () =>
    assert.equal(await page.text("#attention-output"), "[[3.401],[3.604]]"));

  await check("next walks all three steps and stops at the end", async () => {
    await page.click("#att-next");
    assert.equal(await page.text("#att-step-indicator"), "step 2 / 3");
    assert.equal(await page.text("#att-step-label"), "P = softmax(S)");
    await page.click("#att-next");
    assert.equal(await page.text("#att-step-indicator"), "step 3 / 3");
    assert.equal(await page.eval('document.querySelector("#att-next").disabled'), true);
    await page.click("#att-prev");
    assert.equal(await page.text("#att-step-indicator"), "step 2 / 3");
  });

  await check("arrow keys drive the calculator", async () => {
    await page.focus("#att-prev");
    await page.press("ArrowRight");
    assert.equal(await page.text("#att-step-indicator"), "step 3 / 3");
    await page.press("ArrowLeft");
    assert.equal(await page.text("#att-step-indicator"), "step 2 / 3");
  });

  await check("the live region narrates the step on screen", async () =>
    assert.match(await page.text("#attention-status"), /^Step 2 of 3: row-softmax/));

  // ------------------------------------------ chapter 3: streaming state ---
  section("chapter 3 · online softmax state");

  await check("starts empty: no maximum, no tiles read", async () => {
    assert.equal(await page.text("#state-tile"), "—");
    assert.equal(await page.text("#state-max"), "−∞");
    assert.equal(await page.text("#state-norm"), "0");
    assert.equal(await page.text("#state-progress-text"), "0 / 2");
    assert.match(await page.text("#state-status"), /Start with an empty state/);
  });

  await check("one step reads two of the three keys", async () => {
    assert.equal(await page.count("#state-cells .state-cell"), 3);
    await page.click("#state-step");
    assert.equal(await page.text("#state-tile"), "1 / 2");
    assert.equal(await page.text("#state-max"), "0.707");
    assert.equal(await page.text("#state-merge-rule"), "initialize state");
  });

  await check("the finished row equals full softmax exactly", async () => {
    await page.click("#state-step");
    assert.equal(await page.text("#state-progress-text"), "2 / 2");
    assert.equal(await page.text("#state-output"), "3.401");
    assert.match(await page.text("#state-status"), /Complete: O = 3\.401/);
    await page.click("#state-reset");
    assert.equal(await page.text("#state-progress-text"), "0 / 2");
  });

  await check("a rising maximum forces the rebase branch", async () => {
    await page.click('[data-state-demo="new-max"]');
    assert.equal(await page.eval('document.querySelector(\'[data-state-demo="new-max"]\').getAttribute("aria-pressed")'), "true");
    assert.equal(await page.text("#state-max"), "−∞");
    await page.click("#state-step");
    await page.click("#state-step");
    assert.equal(await page.text("#state-max"), "3.536");
    assert.equal(await page.text("#state-merge-rule"), "rebase + add");
  });

  await check("play runs to the end, then releases the button", async () => {
    await page.click('[data-state-demo="same-max"]');
    await page.click("#state-play");
    assert.equal(await page.eval('document.querySelector("#state-play").getAttribute("aria-pressed")'), "true");
    assert.equal(await page.eval('document.querySelector("#state-play").disabled'), true);
    assert.equal(await page.text("#state-tile"), "1 / 2", "play must show the first tile at once");
    await sleep(2100);
    assert.equal(await page.text("#state-progress-text"), "2 / 2");
    assert.equal(await page.eval('document.querySelector("#state-play").getAttribute("aria-pressed")'), "false");
    assert.equal(await page.eval('document.querySelector("#state-play").disabled'), false);
  });

  await check("space steps and r resets, from the keyboard", async () => {
    await page.click("#state-reset");
    await page.focus("#state-step");
    await page.press(" ");
    assert.equal(await page.text("#state-tile"), "1 / 2");
    await page.press("r");
    assert.equal(await page.text("#state-tile"), "—");
  });

  // --------------------------------------------- chapter 4: byte counter ---
  section("chapter 4 · tile and HBM byte counter");

  await check("the defaults N=4096 d=128 give the documented split", async () => {
    assert.equal(await page.text("#naive-total"), "132 Mi");
    assert.equal(await page.text("#fa-total"), "4 Mi");
    assert.equal(await page.text("#traffic-ratio"), "33×");
  });

  await check("both traffic bars are drawn with real segments", async () => {
    assert.equal(await page.count("#bar-naive span"), 3);
    assert.equal(await page.count("#bar-fa span"), 2);
    assert.match(await page.text("#tile-status"), /memory-bound/);
  });

  await check("a longer sequence grows the saving", async () => {
    await page.setRange("#sequence-length", 32768);
    assert.equal(await page.text("#sequence-value"), "32 768");
    assert.notEqual(await page.text("#naive-total"), "132 Mi");
    assert.equal(await page.eval(`(() => {
      const ratio = document.querySelector("#traffic-ratio").textContent;
      return Number(ratio.replace(/[^0-9.]/g, ""));
    })()`) > 33, true, "traffic ratio should climb past the N=4096 value");
  });

  await check("the head dimension moves the intensity, not just the bytes", async () => {
    await page.setRange("#sequence-length", 4096);
    await page.setRange("#head-dim", 256);
    assert.equal(await page.text("#head-dim-value"), "256");
    assert.match(await page.text("#intensity-fa"), /FLOP\/B$/);
    await page.setRange("#head-dim", 128);
  });

  // ------------------------------------------------ chapter 5: work map ---
  section("chapter 5 · work partitioning");

  await check("opens on FA-1, four warps, cross-warp reduction", async () => {
    assert.match(await page.text("#work-status"), /^FA-1 sliced-K/);
    assert.equal(await page.count("#work-diagram .work-cell.fa1"), 4);
    assert.equal(await page.text("#work-traffic-stat"), "yes (shared-mem reduce)");
    assert.equal(await page.eval('document.querySelector(\'[data-work-mode="fa1"]\').getAttribute("aria-pressed")'), "true");
  });

  await check("switching to FA-2 shares one K/V tile and drops the reduction", async () => {
    await page.click('[data-work-mode="fa2"]');
    assert.match(await page.text("#work-status"), /^FA-2 sliced-Q/);
    assert.equal(await page.count("#work-diagram .work-cell.fa2"), 4);
    assert.equal(await page.text("#work-traffic-stat"), "none");
    assert.equal(await page.text("#work-output-stat"), "independent per warp");
    assert.equal(await page.eval('document.querySelector(\'[data-work-mode="fa2"]\').getAttribute("aria-pressed")'), "true");
    assert.equal(await page.eval('document.querySelector(\'[data-work-mode="fa1"]\').getAttribute("aria-pressed")'), "false");
  });

  // ------------------------------------------------- progress rail ---
  section("reading progress rail");

  await check("opens on chapter 1 of 8", async () => {
    assert.equal(await page.text("#rail-progress-text"), "Chapter 01 / 08");
    assert.equal(await page.eval('document.querySelector(".rail nav a").classList.contains("active")'), true);
  });

  await check("scrolling to chapter 4 moves the marker and remembers it", async () => {
    await page.scrollToSection("flash2");
    assert.equal(await page.text("#rail-progress-text"), "Chapter 04 / 08");
    assert.equal(await page.eval('document.querySelector(".rail nav a[href=\'#flash2\']").hasAttribute("aria-current")'), true);
    assert.equal(await page.eval(`localStorage.getItem("fa2-guide-last-chapter")`), "flash2");
  });

  await check("a first visit offers no resume link, and hides it for real", async () => {
    await page.eval(`localStorage.clear()`);
    await page.goto(server.url);
    const state = await page.eval(`(() => {
      const link = document.querySelector("#resume-reading");
      return { hidden: link.hidden, display: getComputedStyle(link).display };
    })()`);
    assert.equal(state.hidden, true, "no saved position means no resume link");
    assert.equal(state.display, "none", "`hidden` must survive the button's own display rule");
  });

  await check("a return visit offers to resume where the reader stopped", async () => {
    await page.eval(`localStorage.setItem("fa2-guide-last-chapter", "flash2")`);
    await page.goto(server.url);
    const resume = await page.eval(`(() => {
      const link = document.querySelector("#resume-reading");
      return { hidden: link.hidden, display: getComputedStyle(link).display, href: link.getAttribute("href"), text: link.textContent.trim() };
    })()`);
    assert.equal(resume.hidden, false);
    assert.notEqual(resume.display, "none");
    assert.equal(resume.href, "#flash2");
    assert.match(resume.text, /Resume chapter 04/);
  });

  // ------------------------------------------------ speedup calculator ---
  section("speedup calculator");

  await check("the N=4096 default is already capped at 5×", async () => {
    assert.equal(await page.text("#speedup-result"), "5.0×");
    assert.equal(await page.text("#speedup-regime"), "memory-bound");
    assert.match(await page.text("#speedup-status"), /Capped at 5×/);
  });

  await check("the 512 preset lands in the compute-bound regime", async () => {
    await page.click("#speedup-short");
    assert.equal(await page.text("#speedup-n"), "512");
    assert.equal(await page.text("#speedup-result"), "1.1×");
    assert.equal(await page.text("#speedup-regime"), "compute-bound");
  });

  await check("raising batch heads raises occupancy, so the speedup rises", async () => {
    await page.setRange("#speedup-b-input", 1024);
    assert.equal(await page.text("#speedup-result"), "1.2×");
    await page.click("#speedup-short");
  });

  // ------------------------------------------------------ comparator ---
  section("chapter 7 · implementation landscape");

  await check("opens on medium, recommending FlashDecoding for decode", async () => {
    assert.equal(await page.text("#compare-pick"), "FlashDecoding");
    assert.equal(await page.count("#compare-results .compare-row"), 3);
    assert.equal(await page.count("#compare-results .recommended"), 1);
    assert.equal(await page.count('#compare-results [role="listitem"]'), 3);
    assert.match(await page.text("#compare-status"), /^Medium sequences: FlashDecoding wins at 1\.1×\.$/);
  });

  await check("prefill recommends FlashAttention-2 itself", async () => {
    await page.click('[data-compare-mode="short"]');
    assert.equal(await page.text("#compare-pick"), "FlashAttention-2");
    assert.equal(await page.count("#compare-results .recommended"), 1);
  });

  await check("the approximate option is flagged as not exact", async () => {
    await page.click('[data-compare-mode="extreme"]');
    assert.equal(await page.text("#compare-pick"), "Linear attention");
    assert.equal(await page.text("#compare-exact"), "No");
    await page.click('[data-compare-mode="medium"]');
  });

  // --------------------------------------------------- backward pass ---
  section("chapter 5 · backward pass");

  await check("draws all eight steps and starts unplayed", async () => {
    assert.equal(await page.count("#backward-flow .pipeline-step"), 8);
    assert.equal(await page.text("#backward-step"), "—");
    assert.match(await page.text("#backward-status"), /^Start: gradients flow/);
  });

  await check("stepping marks each cell done behind the cursor", async () => {
    await page.click("#backward-step-btn");
    assert.equal(await page.text("#backward-step"), "1 / 8");
    assert.equal(await page.count("#backward-flow .pipeline-step.done"), 1);
    assert.equal(await page.count("#backward-flow .pipeline-step.active"), 1);
  });

  await check("the gradient write is called out on step 4", async () => {
    for (let i = 0; i < 3; i++) await page.click("#backward-step-btn");
    assert.equal(await page.text("#backward-step"), "4 / 8");
    assert.equal(await page.text("#backward-gradient"), "writes ∂L/∂Q or ∂L/∂K or ∂L/∂V");
    assert.equal(await page.text("#backward-accumulate"), "Write to HBM");
  });

  await check("the last step reports the whole backward pass complete", async () => {
    for (let i = 0; i < 4; i++) await page.click("#backward-step-btn");
    assert.equal(await page.text("#backward-step"), "8 / 8");
    assert.equal(await page.count("#backward-flow .pipeline-step.done"), 8);
    assert.match(await page.text("#backward-status"), /^Complete: all gradients computed/);
  });

  await check("play starts, disables itself, and reset stops it", async () => {
    await page.click("#backward-play-btn");
    assert.equal(await page.eval('document.querySelector("#backward-play-btn").getAttribute("aria-pressed")'), "true");
    assert.equal(await page.eval('document.querySelector("#backward-play-btn").disabled'), true);
    assert.equal(await page.text("#backward-step"), "1 / 8");
    await sleep(1200);
    assert.equal(await page.text("#backward-step"), "2 / 8", "the timer should still be advancing");
    await page.click("#backward-reset-btn");
    assert.equal(await page.text("#backward-step"), "—");
    assert.equal(await page.eval('document.querySelector("#backward-play-btn").getAttribute("aria-pressed")'), "false");
    assert.equal(await page.eval('document.querySelector("#backward-play-btn").disabled'), false);
    await sleep(1200);
    assert.equal(await page.text("#backward-step"), "—", "reset must clear the timer");
  });

  // ------------------------------------------------------ hero sweep ---
  section("hero · forward-pass sweep");

  await check("the sweep canvas is drawn at the hero's size", async () => {
    const box = await page.eval(`(() => {
      const c = document.querySelector("#hero-sweep-canvas");
      const r = c.getBoundingClientRect();
      return { w: Math.round(r.width), h: Math.round(r.height), bw: c.width, bh: c.height };
    })()`);
    assert.ok(box.w > 200, `canvas should have real width, got ${box.w}`);
    // device pixels track CSS pixels, so the canvas is not upscaled or blurred
    assert.ok(Math.abs(box.bw - box.w) <= 2, `backing store ${box.bw} vs layout ${box.w}`);
    assert.ok(box.bh > box.bw / 2 && box.bh < box.bw * 2, `unexpected aspect: ${box.bw}x${box.bh}`);
  });

  await check("it actually paints, and the caption advances", async () => {
    // The sweep holds on the finished pass for a beat, so two samples a fixed
    // distance apart can legitimately read the same. Poll instead.
    const seen = new Set();
    for (let i = 0; i < 20; i++) {
      seen.add(await page.text("#hero-sweep-count"));
      await sleep(200);
    }
    assert.ok(seen.size >= 2, `the counter never moved, stuck on "${[...seen][0]}"`);
    for (const value of seen) assert.match(value, /^[1-7] \/ 7 tiles$/);
    const painted = await page.eval(`(() => {
      const c = document.querySelector("#hero-sweep-canvas");
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let lit = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 0) lit++;
      return lit;
    })()`);
    assert.ok(painted > 5000, `expected ink on the canvas, found ${painted} lit pixels`);
  });

  await check("reduced motion freezes it on the finished frame", async () => {
    await page.setReducedMotion(true);
    await sleep(700);
    const still = await page.eval(`(() => {
      const c = document.querySelector("#hero-sweep-canvas");
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let hash = 0;
      for (let i = 3; i < d.length; i += 400) hash = (hash + d[i] * (i % 97)) % 1000003;
      return { hash, count: document.querySelector("#hero-sweep-count").textContent };
    })()`);
    await sleep(700);
    const again = await page.eval(`(() => {
      const c = document.querySelector("#hero-sweep-canvas");
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let hash = 0;
      for (let i = 3; i < d.length; i += 400) hash = (hash + d[i] * (i % 97)) % 1000003;
      return hash;
    })()`);
    assert.equal(still.count, "7 / 7 tiles", "a still frame should show the finished pass");
    assert.equal(again, still.hash, "the canvas must not repaint under reduced motion");
    await page.setReducedMotion(false);
  });

  await check("the sweep carries a description for screen readers", async () => {
    const label = await page.eval(`(() => {
      const c = document.querySelector("#hero-sweep-canvas");
      return { role: c.getAttribute("role"), label: (c.getAttribute("aria-label") || "").length };
    })()`);
    assert.equal(label.role, "img");
    assert.ok(label.label > 60, `aria-label too short to be useful: ${label.label} chars`);
  });

  // -------------------------------------------------- accessibility ---
  section("accessibility, checked for real");

  await check("every button has a name a screen reader can read", async () => {
    const nameless = await page.eval(`[...document.querySelectorAll("button")]
      .filter(b => !(b.textContent.trim() || b.getAttribute("aria-label") || b.getAttribute("title")))
      .map(b => b.id || b.className || b.outerHTML.slice(0, 60))`);
    assert.deepEqual(nameless, []);
  });

  await check("every input has a label, aria-label, or aria-labelledby", async () => {
    const unlabelled = await page.eval(`[...document.querySelectorAll("input")]
      .filter(el => {
        if (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby")) return false;
        return !(el.id && document.querySelector('label[for="' + CSS.escape(el.id) + '"]'));
      })
      .map(el => el.id || el.type)`);
    assert.deepEqual(unlabelled, []);
  });

  await check("toggles report their pressed state, not just a colour", async () => {
    const undeclared = await page.eval(`[...document.querySelectorAll("[aria-pressed]")]
      .filter(el => !["true", "false"].includes(el.getAttribute("aria-pressed")))
      .map(el => el.id || el.dataset.stateDemo || el.dataset.workMode || el.dataset.compareMode)`);
    assert.deepEqual(undeclared, []);
  });

  await check("the skip link is off screen until it takes focus", async () => {
    const hidden = await page.eval('(() => { const r = document.querySelector(".skip-link").getBoundingClientRect(); return Math.round(r.bottom); })()');
    assert.ok(hidden <= 0, `skip link should sit above the viewport, bottom was ${hidden}`);
    await page.focus(".skip-link");
    const shown = await page.eval('(() => { const r = document.querySelector(".skip-link").getBoundingClientRect(); return Math.round(r.top); })()');
    assert.ok(shown >= 0, `focused skip link should be visible, top was ${shown}`);
    await page.eval('document.querySelector(".skip-link").blur()');
  });

  await check("reduced motion really removes the transitions", async () => {
    const selector = ".work-cell";
    const longest = value => value.split(",").map(v => parseFloat(v)).reduce((a, b) => Math.max(a, b), 0);
    await page.setReducedMotion(false);
    const normal = await page.eval(`getComputedStyle(document.querySelector(${JSON.stringify(selector)})).transitionDuration`);
    await page.setReducedMotion(true);
    const reduced = await page.eval(`getComputedStyle(document.querySelector(${JSON.stringify(selector)})).transitionDuration`);
    assert.ok(longest(normal) > 0.1, `${selector} should transition normally, got ${normal}`);
    assert.ok(longest(reduced) < 0.001, `expected ~0s under reduce, got ${reduced}`);
    await page.setReducedMotion(false);
  });
} finally {
  await page.close();
  server.close();
}

console.log(`\nbrowser checks: ${ran - failures}/${ran} passed`);
process.exit(failures ? 1 : 0);