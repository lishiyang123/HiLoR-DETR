# BTC runtime reference

BTC and CDW1630 were executed in the same recorded DEIM runtime. The files in
this directory are copied from that shared runtime record. Dataset-specific
paths are supplied by the BTC materialized configs and are not environment
dependencies. GPU/CUDA driver differences can prevent checkpoint bytes from
being bit-identical across machines; reproduction is judged by the documented
protocol and close metrics, while packaged assets are checked by SHA256.
