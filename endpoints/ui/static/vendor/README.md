# Vendored libraries

Third-party files used by the chat client, copied in unmodified so the client
needs no build step or package manager.

| File | Source | Version | License |
|------|--------|---------|---------|
| `markdown-it.min.js` | `dist/browser/markdown-it.umd.min.js` from the npm package [markdown-it](https://www.npmjs.com/package/markdown-it) | 15.0.2 (npm shasum `5bc092394e9693519f6d5f499d3575469fec62f3`) | MIT, with bundled dependencies under MIT and BSD-2-Clause; see `markdown-it.LICENSE.txt` |

To update, run `npm pack markdown-it@<version>` in an empty folder, check the
tarball's shasum against `npm view markdown-it@<version> dist.shasum`, and copy
`package/dist/browser/markdown-it.umd.min.js` over `markdown-it.min.js`.
Update the version here and the license file if the bundled packages changed.
