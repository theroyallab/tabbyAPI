# Vendored libraries

Third-party files used by the chat client, copied in unmodified so the client
needs no build step or package manager.

| File | Source | Version | License |
|------|--------|---------|---------|
| `markdown-it.min.js` | `dist/browser/markdown-it.umd.min.js` from the npm package [markdown-it](https://www.npmjs.com/package/markdown-it) | 15.0.2 (npm shasum `5bc092394e9693519f6d5f499d3575469fec62f3`) | MIT, with bundled dependencies under MIT and BSD-2-Clause; see `markdown-it.LICENSE.txt` |
| `highlight.js/highlight.min.js`, `highlight.js/languages/{dockerfile,powershell,cmake,latex}.min.js` | `highlight.min.js` (the common languages) and the same files under `languages/` from the npm package [@highlightjs/cdn-assets](https://www.npmjs.com/package/@highlightjs/cdn-assets) | 11.12.0 (npm shasum `19b1b064bdd3d02705a200e9fa6538d41a566a47`) | BSD-3-Clause; see `highlight.js/LICENSE.txt` |
| `katex/katex.min.js`, `katex/katex.min.css`, `katex/fonts/*.woff2` | The same files under `dist/` from the npm package [katex](https://www.npmjs.com/package/katex) (only the WOFF2 fonts, which every supported browser picks first) | 0.19.0 (npm shasum `b69c8cbcfc8c331be3f4f6e606b02b7146501ec7`) | MIT; see `katex/LICENSE.txt` |

To update one, run `npm pack <package>@<version>` in an empty folder, check the
tarball's shasum against `npm view <package>@<version> dist.shasum`, and copy
the files listed above over the ones here (for markdown-it,
`package/dist/browser/markdown-it.umd.min.js` becomes `markdown-it.min.js`).
Update the version and shasum here, and the license file if it changed.

The client loads highlight.js and KaTeX only when a reply has a code block with
a language or some math in it.
