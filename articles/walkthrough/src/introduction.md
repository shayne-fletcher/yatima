# The maintainer walkthrough

This is the map back into Yatima: how a prompt becomes model output, capability-bounded action, and a view event, with each boundary checked against the laws that justify it.

The chapters follow the implementation in dependency order, beginning with the overall map and then opening each part of the source in turn. They describe commit `402d34a26bedd9d254e76a51be8c034961e28db1`, including the Candle and managed `llama-server` backends, Muse Glimmer's ATEM protocol, and the native and browser frontends built over the shared host.
