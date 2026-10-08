# CLAUDE.md

Static, dependency-free teaching site for FlashAttention-2. No build step, no npm install, no framework. Live at https://atandra2000.github.io/flash-attention-2-guide/

## Commands

```bash
npm test                    # test-guide.mjs + test-browser.mjs
npm run test:static         # content checks only
npm run test:browser        # headless browser checks
npm run serve               # python3 -m http.server 8137
```

## Layout

| Path | Holds |
|---|---|
| `index.html` | Page shell |
| `app.js` | All interactive behaviour |
| `styles.css` | All styling |
| `docs/` | Chapter content |
| `test-guide.mjs` | Content and structure checks |
| `test-browser.mjs` | Browser behaviour checks |

## Notes

- Keep the site dependency-free. Do not add npm packages or a bundler.
- Both tests must pass before you call a change done.
- Use `codegraph explore "<query>"` before you grep. The index is in `.codegraph/`.
