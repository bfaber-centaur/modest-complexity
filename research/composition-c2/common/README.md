# c2 shared code

Pinned by the coordinator (DECISIONS D3); lanes do not edit it. Conventions, parameter counts and test list: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md) D3.

| file | contents |
|---|---|
| `c2task.py` | r1's task (as `c2task.r1`), task/split IDs, rows of any program length, N/notN held-out rows |
| `c2rnn.py` | RNN with tied or per-step transition and readout groups, any length, continuation from a hidden state, loss/gradient with explicit step weights |
| `c2train.py` | one run or one exact continuation (`--resume`, full Adam state) on length-3 programs; r1-compatible metrics plus N/notN and `--diag` |
| `c2probe.py` | decoder-across-position table (D5) |
| `test_c2.py` | 10 tests, own process: `uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py` |

Lane scripts put this directory and `research/regimes/common/` on `sys.path` (as `c2train.py` does) and import `c2task`, `c2rnn`, `c2probe`, `nets`.

```bash
export PATH=~/.local/bin:$PATH; uv sync --locked --group analysis
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/<lane>/plan.json --root runs/c2 --registry research/composition-c2/<lane>/registry.jsonl --session-label c2-<lane>-cloud-1"
$R --workers 2 --ids <run IDs>
```
