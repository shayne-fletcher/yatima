# Relevant research

Work that informs yatima's shape — language-integrated model calls, local
inference, and grounded/auditable output.

- Anil Madhavapeddy, [*Language Integrated LLMs as an OCaml Function*](https://anil.recoil.org/notes/language-integrated-llms).
  The kindred idea: treat a model call as an ordinary typed function in the host
  language rather than a service boundary. yatima follows it in Rust.

- NVIDIA, [*SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness*](https://arxiv.org/abs/2609.20519) (arXiv 2609.20519, 2026). A local reading copy lives in the untracked `papers/` directory. An AI reads agent traces, proposes harness changes, and keeps a change only if it passes two gates fixed before the search began: capability stays within a declared tolerance, and an efficiency metric improves. Frozen candidates are evaluated once on held-out tasks, and held-out results never feed back into the search. The paper's point for yatima is that method, more than the four mechanisms it found (fused edit-and-run actions, cost-gated compaction, handle-based large observations, verified log reduction): a harness tuned against the last trace it saw overfits that trace. Its related work also points at evidence that evolved harnesses overfit their search tasks (arXiv 2607.12227) and at moving deterministic control flow out of the model and into code (LLM-as-Code, arXiv 2606.15874).
