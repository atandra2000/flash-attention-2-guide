import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  stableSoftmax,
  standardAttention,
  streamingAttention,
  calculateNaiveBytes,
  calculateFABytes,
  calculateSpeedup,
  isSpeedupCapped,
  _internals,
} from "./app.js";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
const app = readFileSync(new URL("./app.js", import.meta.url), "utf8");
for (const id of ["attention", "memory-wall", "online-softmax", "flash2", "kernel", "extensions", "comparison", "foundations"]) {
  assert.match(html, new RegExp(`id=["']${id}["']`), `missing #${id}`);
}
for (const id of ["attention-calc", "state-step", "tile-size", "work-map", "backward-flow", "state-old-scale", "state-new-mass", "state-merge-rule"]) {
  assert.match(html, new RegExp(`id=["']${id}["']`), `missing #${id}`);
}
assert.doesNotMatch(html, /https?:\/\//, "guide must not fetch remote assets");
for (const phrase of [
  "Scaled dot-product attention", "online softmax", "HBM", "shared memory",
  "exact", "sliced-Q", "MQA", "GQA", "Backward pass"
]) {
  assert.match(html, new RegExp(phrase, "i"), `missing lesson concept: ${phrase}`);
}
assert.match(css, /prefers-reduced-motion/, "missing reduced-motion support");
assert.match(css, /:focus-visible/, "missing visible keyboard focus");
assert.match(html, /class=["']skip-link["']/, "missing skip link");
assert.match(html, /class=["']hero-cta["'][^>]*href=["']#attention["']/, "hero needs a clear chapter-one start action");
assert.match(html, /id=["']resume-reading["']/, "guide should offer a saved-reading resume action");
for (const id of ["rail-progress-text", "rail-progress-fill"]) {
  assert.match(html, new RegExp(`id=["']${id}["']`), `missing progress rail field: ${id}`);
}
assert.doesNotMatch(html, />loading</, "widgets must have readable static fallbacks");
assert.match(html, /id=["']attention-output["'][^>]*>\[\[3\.401\], \[3\.604\]\]</, "attention fallback must match the demonstrated output");
assert.match(html, /id=["']state-play["'][^>]*aria-pressed=["']false["']/, "play control needs an initial state");
assert.match(app, /play\.disabled = true/, "playback must prevent overlapping intervals");
assert.match(app, /index = 0;\n    step\(\);/, "playback must show the first tile immediately");
assert.doesNotMatch(app, /cell\.innerHTML/, "work-map labels must use text nodes");
assert.match(css, /min-height: 44px/, "interactive controls need touch-safe height");
assert.match(css, /scroll-margin-top/, "chapter anchors must clear the sticky navigation");
assert.match(app, /fa2-guide-last-chapter/, "reading location should persist between visits");
for (const concept of ["Start one step earlier", "Follow one score tile’s lifetime", "What the three numbers remember"]) {
  assert.match(html, new RegExp(concept), `missing from-scratch concept bridge: ${concept}`);
}

const probabilities = stableSoftmax([1000, 1001]);
assert.ok(Math.abs(probabilities[0] + probabilities[1] - 1) < 1e-12, "softmax must normalize");
assert.ok(probabilities[1] > probabilities[0], "stable softmax must preserve ordering");
const q = [[1, 0], [0, 1]];
const k = [[1, 0], [0, 1], [1, 1]];
const v = [[2], [3], [5]];
const reference = standardAttention(q, k, v).output;
const streamed = streamingAttention(q, k, v, 2).output;
for (let i = 0; i < reference.length; i++) {
  assert.ok(Math.abs(reference[i][0] - streamed[i][0]) < 1e-10, "streamed attention must match reference attention");
}
assert.match(html, /id=["']sequence-length["'][^>]*type=["']range["']/, "missing labelled tile slider");
for (const id of ["tile-status", "work-status"]) {
  assert.match(html, new RegExp(`id=["']${id}["'][^>]*aria-live=["']polite["']`), `missing announced status: ${id}`);
}

assert.ok(calculateNaiveBytes(4096, 128) > calculateFABytes(4096, 128), "naive bytes must exceed FA bytes");
assert.equal(calculateNaiveBytes(2, 2), 2 * (3 * 2 * 2 + 4 * 2 * 2 + 2 * 2), "naive byte formula closed form");
assert.equal(calculateFABytes(2, 2), 2 * (3 * 2 * 2 + 2 * 2) + 2 * 2 * 4, "FA byte formula closed form");
const ratio2048 = calculateNaiveBytes(2048, 128) / calculateFABytes(2048, 128);
const ratio4096 = calculateNaiveBytes(4096, 128) / calculateFABytes(4096, 128);
assert.ok(ratio4096 > ratio2048, "IO reduction grows with N");
assert.ok(calculateSpeedup(512, 64) > 1.0, "speedup must exceed 1×");
assert.ok(calculateSpeedup(8192, 64) > calculateSpeedup(512, 64), "speedup grows with N");
assert.ok(calculateSpeedup(4096, 64) <= 5.0, "speedup respects ceiling");
assert.equal(typeof isSpeedupCapped(8192, 64), "boolean", "isSpeedupCapped returns boolean");
assert.equal(_internals.formatBytes(2048), "2 Ki", "formatBytes unit math");
assert.equal(_internals.formatBytes(2 * 1024 * 1024), "2 Mi", "formatBytes Mi math");
assert.equal(_internals.formatBytes(3 * 1024 ** 3), "3 Gi", "formatBytes Gi math");

console.log("guide static checks passed");
