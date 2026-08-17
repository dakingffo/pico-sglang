"""MTPGraph — CUDA-graph batch draft path for the MTP drafter. DEFERRED.

Not implemented yet (milestone order: eager sequential exact-port first, graph after the
greedy-consistency milestone passes end-to-end). When implemented, this mirrors
``engine/graph.py``'s ``GraphRunner`` (static buffers, bs buckets ``[1,2,4,8,...,max]``,
descending capture + shared pool, ``pad_batch`` with dummy rows):

- step-0 graph : (bs_padded, L_max) — carry window padded to the static max L, static KV,
  attention-mask padding.
- steps 1..K-1 graph : (bs_padded, 1) — single-row autoregressive, static-growing KV.

bs bucket params come straight from the server startup args (``cuda_graph_bs`` /
``cuda_graph_max_bs``), independent of whether the main model's hybrid can use graphs.
``cuda_graph_bs=[]`` or ``max_bs=0`` falls back to the eager ``MTPEngine``.
"""
