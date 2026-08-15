# Walkthrough registry audit

This is the verification companion to the maintainer walkthrough. It records how authors and reviewing agents check law identifiers, witnesses, and eventual chapter coverage. It is not part of the runtime reading path: the chapters explain Yatima's code, while this document checks that their claims remain tied to the source.

The audit describes commit `d724b9ed2f07b709dff29597ff91f24aff5ac8ad`.

## Audit record

The source already names its important obligations. The audit treats each entry as a record:

```text
Law     = Id x Scope x Statement x NonEmpty<Witness>
Witness = Static(Type | Module) + Dynamic(Test) + Procedural(Audit)
```

Here `=` means "is defined as," `x` combines fields that are all present, `+` separates alternatives, and `|` separates alternatives nested inside `Static`. `NonEmpty<Witness>` is a nonempty collection of witnesses. Read in full: every law has an id, scope, statement, and at least one witness; every witness is static type or module evidence, a dynamic test, or a procedural audit. One law may need several witnesses, and one test may support several laws.

A static witness uses types or module visibility to prevent an invalid construction or dependency. A dynamic witness runs a test. A procedural witness gives a repeatable review step for a rule Rust cannot enforce. A comment beside an implementation is not a test, and a test of one example does not necessarily prove a rule about the whole design.

## Global checks

For each source registry `R_i`, the global registry is:

```text
R = union_i R_i
```

Here `R_i` names registry number `i`, and `union_i` combines every such registry. In plain English, `R` contains all laws declared anywhere in the source tree.

Well-formedness requires:

```text
for all l1, l2 in R,
    id(l1) = id(l2)  implies  l1 = l2

for all l in R,
    witnesses(l) is nonempty

for all w in witnesses(l),
    resolve(pinned_commit, w) is defined
```

This is quantified pseudocode: `l`, `l1`, and `l2` stand for arbitrary laws, while `w` stands for an arbitrary witness attached to a law. `implies` states a required consequence, and `resolve(pinned_commit, w) is defined` means that the cited evidence can be found at that commit. The three conditions say that one id cannot name two different laws, every law has at least one witness, and every witness is findable.

The first condition can be summarized in mathematical function notation as `Id -> Law`: each identifier selects at most one law. It is stronger than "each file has no duplicate" because identifier sets must be disjoint across the whole union. A witness that cannot be found at the pinned commit is not evidence.

## Preflight inventory

The preflight searched registry-form module documentation and example-level registry documentation, then located `upholds:` citations and non-test witnesses in the pinned tree. It found eight registry homes, not one. In the `Ids` column below, a prefix followed by `*` means every id with that prefix, while a range such as `CLI-1..3` means `CLI-1` through `CLI-3`:

| Home | Records | Distinct ids | Ids |
|---|---:|---:|---|
| `lib/src/lib.rs` | 58 | 57 | `LAYER-*`, `MS-*`, `MD-*`, `EOS-*`, `FETCH-*`, `MEM-*`, `SAM-*`, `STOP-*`, `GEN-*`, `GE-*`, `ARCH-*`, `PREFILL-*`, `FMT-*`, `CAPS-*`, `PROFILE-*`, `CTX-*`, `COMPACT-*`, `RT-*`, `CMP-*`, `AGENT-*`, `TOOL-*`, `CAP-*`, `WIN-*`, `PLOT-*`, `IMG-*`, `PROTO-1`, `OBS-*`, `TMPL-*`, `REASON-*`, `CHAT-*` |
| `lib/examples/investment_thesis.rs` | 2 | 2 | `COMPARE-1`, `META-1` |
| `cli/src/main.rs` | 3 | 3 | `CLI-1..3` |
| `host/src/lib.rs` | 5 | 5 | `HOST-1..5` |
| `protocol/src/lib.rs` | 2 | 2 | `PROTO-2`, `WASM-1` |
| `serve/src/lib.rs` | 3 | 3 | `SRV-1..3` |
| `tui/src/lib.rs` | 7 | 7 | `TUI-1..7` |
| `web/src/lib.rs` | 7 | 7 | `WEB-1..7` |
| **Union** | **87** | **86** | one collision |

`yatima-gui` and `yatima-text` have no local registry. That is not the same as having no obligations. The GUI contains dynamic witnesses for inherited `IMG-1` and `HOST-4`; `yatima-text` participates in the WASM-clean dependency base exercised through the web build. Their laws remain owned by the registry that states them.

## Identifier result

The union is not well formed. `FETCH-1` denotes two different laws in the same library registry:

1. after model download, re-check completeness before load ([`lib/src/lib.rs:33`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/lib.rs#L33));
2. within a session, fetch each resolved `ReadPage` URL at most once ([`lib/src/lib.rs:180`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/lib.rs#L180)).

Therefore identifier-to-law lookup is not a function. The nearby design prose already calls the model-completeness law `FETCH-2`, which is the minimal rename to make in a separate source commit.

## Witness result

Seventy-three of the 86 distinct ids have at least one findable test carrying an `upholds:` citation. Five other ids have honest non-test witness kinds:

| Id | Kind | Findable witness |
|---|---|---|
| `LAYER-1` | procedural | audit the module dependency order declared in [`notes/design.md`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/notes/design.md#L112) and the crate manifests |
| `CMP-1` | static | native `async fn` and generic-only boundary in [`completer.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/completer.rs#L80) |
| `HOST-1` | procedural | audit frontend decode imports/calls; the registry itself calls this grep-enforced review |
| `HOST-3` | static/module | `HostConfig` crosses the thread; `Engine` is constructed and retained inside `actor_main` ([`host/src/lib.rs:147`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/host/src/lib.rs#L147)) |
| `WASM-1` | procedural | run [`scripts/check-wasm.sh`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/scripts/check-wasm.sh); CI repeats the compiler audit on every push |

The remaining eight ids have a witness-index defect:

| Id | What exists | Defect |
|---|---|---|
| `MEM-1`, `MEM-2` | memory predicate and RSS tests in `engine.rs` | tests do not cite the ids with `upholds:` |
| `TOOL-1` | watch/join and cancellation tests in `tool.rs` | tests cite neighboring agent/capability laws, not `TOOL-1` |
| `TOOL-2` | `tool_outcome_projects_to_model_result` | the projection test has no law citation |
| `TUI-2` | the immutable `ui(&App)` rendering boundary | no test citation despite the TUI registry's claim that every law has one |
| `META-1` | `print_run_metadata` implementation | no test citation despite the example registry's claim that its laws are tested |
| `OBS-3` | `.instrument(span)` at the spawn site | its sole `upholds:` citation is an implementation comment, not a test |
| `WEB-2` | the WASM submit gate | its sole `upholds:` citation is an implementation comment, not a test |

This is not a demand for eight new tests. The correction may be to add a missing citation to an existing test, add a compile-time or procedural witness, or narrow an overstated registry preamble. The chapter owning each law must make that choice after reading the implementation.

## Completion check

Each operational chapter includes only the laws encountered while explaining its code. At the end, the walkthrough must account for the whole global registry:

```text
union_chapter laws(chapter) = R
```

Here `laws(chapter)` is the set of laws covered by one chapter, and `union_chapter` combines those sets for every chapter. The equation says that the completed walkthrough must cover every law in the global registry.

## Audit debts

1. Rename the model-completeness `FETCH-1` to `FETCH-2` in a separate source commit and update its witnesses and prose references.
2. Resolve the eight witness-index defects above in the chapter that owns each law; do not manufacture tests solely for visual uniformity.
3. Reconcile registry preambles that promise a citing test for every law with the accepted static and procedural witness alternatives.
4. Extend `lib/examples/invariant_reviewer.rs`: it currently extracts ids only from `lib/src/lib.rs` and `cli/src/main.rs`, and scans `upholds:` only below `lib/src` and `cli/src`. It cannot review the global registry it now claims to support.
5. Decide whether the example-level `COMPARE-1`/`META-1` registry remains an auxiliary registry or moves to a canonical crate-level home. Until then, the global union must include it explicitly.
