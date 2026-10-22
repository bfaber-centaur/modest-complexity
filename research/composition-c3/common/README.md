# c3 shared code

Coordinator-owned, pinned in [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md) D3. It imports c2's pinned `../../composition-c2/common/` (`b3e8132`) and r1's `research/regimes/common/` unchanged.

| file | what it is |
|---|---|
| `c3train.py` | c2train.py with a step-weight schedule (`--switch-step`, `--switch-weights`, `--taper-updates`) and a recipe check on `--resume` (`--resume-config`). Without a switch it is c2train.py bitwise. |
| `c3correct.py` | `corrected_forward(p, s, ops, depth, correct)`: the frozen base to depth d, a correction applied to h_d alone, then the unchanged remaining steps. Also `frozen(p)` and `params_sha256(p)`. |
| `test_c3.py` | 8 checks; run in its own process. |

## Commands

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23
uv run --locked python -m unittest -v research/composition-c3/common/test_c3.py   # own process
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py   # own process
```

Branch from a full-state checkpoint (here, a drop of the intermediate losses at 10k):

```bash
uv run --locked python research/composition-c3/common/c3train.py --out runs/c3/ID --run-id ID \
  --task-id ops51000-ls51100 --transition tied --readout tied --step-weights 1 1 1 --hidden 128 --lr 1e-3 --wd 0.1 \
  --init-seed 51200 --updates 50000 --switch-step 10000 --switch-weights 0 0 1 --taper-updates 1 --diag \
  --resume research/composition-c2/recovered/r1-comp-rnn-inter-s51200-a1/ckpt-10000.npz \
  --resume-sha256 046f4c250d8c7aacc9999437f0223fc029cee6e2cbae6c1644b282e20b9f8010 \
  --resume-config research/regimes/composition/runs/r1-comp-rnn-inter-s51200-a1/config.json
```

The recovered checkpoints' configs are r1's: `research/regimes/composition/runs/<ID>/config.json`, or `research/regimes/review/runs/<ID>/config.json` for the per-step-readout arm. Their hashes are in `../../composition-c2/recovered/RECOVERY.json`.

## Schedule

The update taken at step t uses w(t) = w_b before S. From S on, it uses w_b + min(1, (t − S + 1)/D)·(w_a − w_b). `metrics.jsonl` logs `step_weights` on every row, and `config.json` → `exposure.labels_this_run` / `labels_from_init` gives, per head, the updates with nonzero weight and the weighted row presentations.
