# GPU vs CPU backend benchmark (2026-06-09)

JAX-CUDA was installed in WSL2 (Ubuntu 26.04, RTX 5070 Ti Laptop, CUDA
12.9, jax 0.10.1 with cuda12 plugin). GPU is correctly detected:
`jax.default_backend() == "gpu"`, `jax.devices() == [CudaDevice(id=0)]`.

## Result: GPU is ~2.7x SLOWER than CPU for this workload

Identical workload (10 cont-substrate runs, N=30 agents, T=500 steps):

| Backend | First run (compile+exec) | Steady-state/run | Total (10 runs) |
|---|---|---|---|
| **CPU** (Windows native) | 10.95 s | 9.54 s | 96.8 s |
| **GPU** (WSL2 CUDA) | 28.91 s | 25.65 s | 259.7 s |

Full test suite (105 tests):
- CPU: ~77-99 s
- GPU: 171 s

The GPU is consistently ~2.5-2.7x slower.

## Why: the bottleneck is the Python step-loop, not the FLOPs

`run_cont` (and the other `run_*` substrates) iterate the simulation with
a plain Python `for t in range(n_steps)` loop. Each step:

- dispatches many small JAX ops on tiny arrays (the per-agent state is
  (30, K) with K=2, so (30, 2) and (30, 4)-shaped arrays)
- materializes intermediate results back to host (the step returns an
  `info` dict with `float(...)` conversions and `np.asarray(...)` calls)

On GPU, every one of these tiny ops incurs:
1. kernel launch / dispatch latency (microseconds, but x hundreds of ops
   x 500 steps)
2. host<->device memory transfer for the per-step `info` materialization
   and the `np.asarray` round-trips inside the step

The arrays are far too small to amortize this overhead. The GPU spends
nearly all its time waiting on dispatch and PCIe sync, not computing. On
CPU there is no host-device boundary, so the same tiny ops run with much
lower per-op overhead.

The "steady-state ~= first-run" timing on CPU (9.5 vs 11.0) confirms XLA
compile is NOT the dominant cost -- the Python loop is. GPU compile adds
a few seconds on top but the per-step dispatch is what kills it.

## What this means

The GPU setup is complete and correct, but **does not help the current
code as written**. Hardware is not the bottleneck; code structure is.

## The actual path to speedup (code, not hardware)

To make these sweeps fast -- on CPU first, then with a real GPU win --
the simulation loop needs to be a single compiled kernel:

1. **`jax.lax.scan` over the time loop.** Replace the Python
   `for t in range(n_steps)` in `run_cont` / `run_simple` /
   `run_structural` with a `lax.scan` whose carry is the state and whose
   per-step output is stacked. This compiles all 500 steps into one
   XLA program -- no per-step Python dispatch, no per-step host sync.
   Expected: large speedup on BOTH backends.

2. **`jax.vmap` over the sweep.** Treat the sweep parameters
   (R_in, c0, seed) as a batch dimension. Compile once, run all 75 cells
   in parallel. This is where the GPU finally wins: a (75, 30, 2) batch
   is big enough to saturate the device.

3. **Avoid per-step host materialization.** Accumulate the `info`
   trajectory inside the scan as device arrays; pull to host once at the
   end.

Estimated effort: the scan refactor is the biggest piece (~half a day,
must preserve the trust-update and resource-flow semantics exactly,
re-pass all 105 tests). The vmap-over-sweep is straightforward once the
config values are traceable.

Until that refactor, the fastest way to run sweeps is CPU with the
existing Python-loop code, or CPU with a multiprocessing pool over the
sweep cells (embarrassingly parallel, no code change to the substrate).

## Practical recommendation

- For the sweeps we have now (E11-E17): **run on CPU**. GPU is slower.
- If sweeps grow large enough to matter (1000s of cells), do the
  `lax.scan` + `vmap` refactor; that helps CPU immediately and unlocks
  the GPU.
- The WSL JAX-CUDA environment is preserved at `~/venvs/socialock/` and
  will pay off after the refactor.
