#!/bin/sh
# WASM-1: yatima-protocol compiles for wasm32-unknown-unknown — the
# guarantee that serve's browser client can deserialize the event plane
# without dragging native dependencies. Run locally before touching the
# protocol crate's dependencies; CI runs it on every push.
set -e
rustup target list --installed | grep -q wasm32-unknown-unknown \
  || rustup target add wasm32-unknown-unknown
exec cargo check -p yatima-protocol --target wasm32-unknown-unknown
