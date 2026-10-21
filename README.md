# modest-complexity

Small CPU experiments on the costs and robustness of learned modular-addition implementations.

**Status: instrumented pre-run prototype, not a frozen confirmatory study.** No full pilot or confirmatory run has been performed here. `PROTOCOL.md`, `analysis.py`, and `run_study.py` preserve the original proposal; its composite verdict and top-four-frequency criterion need revision before confirmation. See `PRELAUNCH_REVIEW.txt` and `PRE_RUN.md`.

Python 3.11+; NumPy only. No GPU or dataset download required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python experiment.py --selftest
python -m unittest -v test_instrumentation
python benchmark.py --out runs/bench-serial --steps 1000 --jobs 2 --workers 1
python benchmark.py --out runs/bench-two --steps 1000 --jobs 2 --workers 2
```

Both benchmarks use the same calibration seeds (900000 and 900001), outside the pilot/main ranges. Each child includes all current diagnostics with the normal 32 shams, plus compressed checkpoint and metrics writes. Compare aggregate updates/second and per-phase timings. Default child timeout: 120 seconds. A timeout exits nonzero and preserves partial output; it is not a scientific non-grokking result. Four workers are available for a separately measured comparison, not assumed optimal.

Each run writes `config.json` (configuration, source hashes, Git state, platform, NumPy/BLAS configuration, thread settings, CPU affinity, cgroup evidence and free disk), `runtime.json` (phase wall/CPU time and process peak RSS), `metrics.jsonl`, `data.npz`, and weight/Adam-moment checkpoints. Benchmark parent logs include startup time and exit status. Never overwrite an existing output directory.

Timing separates setup, data writes, training, evaluation, compression, Fourier interventions, readout fitting, checkpoint writes, and metrics writes. Unattributed wall time includes loop overhead and runtime-report writes. Peak RSS is a process-lifetime high-water mark, not whole-job or whole-session memory. Cgroup readings are raw process/ancestor evidence, not a claim of dedicated hardware.

Logged objective summaries are cross-entropy and signed correct-vs-best-incorrect margins for each split. At logging steps, optimizer diagnostics measure the **actual final step's** adaptive Adam displacement and decoupled decay displacement, including their cosine. These are not interval averages or measurements of algorithmic simplicity. Zero-vector cosine is null.

Checkpoints already contain weights, Adam moments and step; **automatic resume is not implemented**. Keep the configuration and data beside them. Interrupted runs must not be silently treated as complete.
