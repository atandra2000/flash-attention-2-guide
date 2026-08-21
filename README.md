# FlashAttention-2 — An Exact Attention Field Guide

A static, dependency-free teaching site that builds FlashAttention-2 from
first principles: why standard attention hits a memory wall, how online
softmax removes it, and what the FlashAttention-2 kernel actually does with
SRAM tiles — forward and backward.

**Read it live:** https://atandra2000.github.io/flash-attention-2-guide/

Part of a from-scratch deep learning portfolio — see
[atandra2000](https://github.com/atandra2000) for the accompanying model
implementations (GPT-2 → FlashAttention-style attention in raw PyTorch).

## What's inside

| Chapter | Concept |
|---|---|
| 01 · Attention | Scaled dot-product attention, worked numerically |
| 02 · Memory wall | HBM vs SRAM, why the O(N²) score matrix is the bottleneck |
| 03 · Online softmax | Streaming softmax — the key trick, derived step by step |
| 04 · FlashAttention-2 | Tiled recomputation, the running (m, l, O) state |
| 05 · Kernel & backward | SRAM tiling, work partitioning, the backward pass |
| 06 · Extensions | MQA, GQA, varlen |
| 07 · Landscape | Where FA-2 sits among attention implementations |
| 08 · Foundations | Prerequisites and further reading |

Interactive widgets: an attention calculator, a step-through tile
visualizer, a memory-byte counter with speedup model, a tile-size slider,
and a backward-pass flow diagram — all with static fallbacks, keyboard
support, and `prefers-reduced-motion` handling. No frameworks, no build
step, no remote assets.

## Run locally

Double-clicking `index.html` won't work — `app.js` is loaded as an ES
module, which browsers block over `file://`. Serve it instead:

```bash
python3 -m http.server 8137
# then open http://127.0.0.1:8137
```

or `npm run serve`.

## Tests

Static checks (structure, accessibility fallbacks, and the math helpers
extracted from `app.js` — softmax stability, streaming-vs-reference
attention parity, IO-byte formulas, speedup bounds):

```bash
npm test        # node test-guide.mjs
```

## Repo layout

```
index.html      # the guide (single page, 8 chapters)
styles.css      # theme + responsive layout
app.js          # widgets + math (ES module, exports tested helpers)
test-guide.mjs  # static + math checks (node:test-style asserts)
```

## License

MIT — see [LICENSE](LICENSE).
