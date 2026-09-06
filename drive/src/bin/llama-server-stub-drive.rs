//! Test wiring only: `CARGO_BIN_EXE_<name>` is set solely for bins of the
//! crate under test, so the driver's hermetic battery needs its own entry
//! to the one llama-server protocol stub. The behavior is single-sourced in
//! `yatima_lib::stub` (the plan's rule: reuse, never duplicate).

fn main() {
    yatima_lib::stub::run()
}
