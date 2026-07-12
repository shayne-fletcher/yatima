#!/bin/sh
# WASM-1: yatima-protocol compiles for wasm32-unknown-unknown — the
# guarantee that serve's browser client can deserialize the event plane
# without dragging native dependencies. Run locally before touching the
# protocol crate's dependencies; CI runs it on every push.
#
# The browser client itself (web/, its own workspace precisely so the
# native `cargo test --workspace` never builds wasm code) checks here too:
# its Cargo.toml is the guard that keeps tokio/candle/yatima-lib out of
# the browser graph, and this is where that guard is exercised.
set -e
rustup target list --installed | grep -q wasm32-unknown-unknown \
  || rustup target add wasm32-unknown-unknown
cargo check -p yatima-protocol --target wasm32-unknown-unknown
cd "$(dirname "$0")/../web"
exec cargo check --target wasm32-unknown-unknown
