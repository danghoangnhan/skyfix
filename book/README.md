# book/

The skyfix Book — an mdBook-rendered tutorial that walks new users through the library.

## Build locally

```sh
cargo install mdbook       # one-time
mdbook serve book/         # live preview at http://localhost:3000
mdbook build book/         # static HTML in book/build/
```

The book defaults to the navy theme and uses MathJax for the few formulas. Source lives in `book/src/`; the table of contents is `book/src/SUMMARY.md`.

## Status

Two chapters are written (trilateration, Bayesian filters). Three placeholders sit in `SUMMARY.md` for the next batch (CRLB, GPU, embedded). All cited empirical numbers come from runnable demos in `crates/skyfix-sim/examples/` and `crates/skyfix-cuda/examples/`.

## Contributing

Each chapter quotes actual working code from the example programs. When you change an algorithm's public API in `skyfix-core`, the corresponding chapter likely needs an update — grep `book/src/` for the API name.
