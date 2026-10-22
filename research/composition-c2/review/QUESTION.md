# c2 independent review: plan, predictions and counterexamples

Role: [../coordinator/charter/06-reviewer.txt](../coordinator/charter/06-reviewer.txt) (Bobby's, verbatim). Decisions: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md).
- **Conventions.** Run IDs `c2-rev-`, seeds 65000–65999, budget 25 min (D6). Files are `rev_*.py` (D3 import hygiene).
- **What I own.** I write only in `research/composition-c2/review/`. I never edit lane or common files.
- **Status values.** PASS, CONCERN, BLOCK or NOT CHECKED. Each names exact artifacts and what it affects.
- **Disclosure.** I wrote r1's per-step-readout arm (`r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`). It is now an input to the time and sharing lanes. I say so wherever a claim I review rests on it, and I checked those files the same way as everyone else's.

## 1. Pin check (`b3e8132`), before any new training

Evidence: [rev_pin.py](rev_pin.py), with output in [out/rev_pin.txt](out/rev_pin.txt). Each item is checked with code written here, not by re-running the pin's own assertions.

| item | how | prediction |
|---|---|---|
| test_c2.py | Run in its own process. | 10/10 pass. |
| Task rows, any length | A plain-Python oracle from the r1 tables (copied as data), lengths 1–6, every row, against `c2task.rows`. Train/held programs and the Latin square, and N/¬N, recomputed independently. | Identical. |
| Init | `c2rnn.init` against the recovered r1 `ckpt-0` arrays: tied/tied against `r1-comp-rnn-inter-s51200`, tied/per-step against my arm. The four schemes compute one function. A copy written in place leaves the other copies unchanged. | Bitwise. |
| Loss and step weights | My own forward and loss, for all four schemes, with weights including 0 and non-unit values. Its value is compared with `c2rnn.loss_grad`, and every gradient entry with central differences of *my* loss. A weight-0 step's untied readout copy gets an exact zero gradient. | Agrees to about 1e-9. |
| Any-length forward and continuation | My forward with an explicit step map at length 5, and a continuation from a mid-program state. | Bitwise. |
| No-change evaluation | All 9 recovered `ckpt-10000`, evaluated with `c2train.evaluate`, against the last published r1 row (r1 keys). | Exactly equal. |
| Recovery record | Every file in `RECOVERY.json`: its committed bytes against the recorded SHA-256 and against the r1 manifest. My three runs are also checked against the originals still on my container. The steps and the scheme stored in each file are checked too. | 27/27. |
| Resume guards | Wrong hash and wrong scheme are refused. *Risk I expect to find:* a resume does not compare `--step-weights`, `--lr`, `--wd`, `--task-id` or `--init-seed` with the source run, so a tied/final run resumed without `--step-weights 0 0 1` would silently become an intermediate-supervised run. | Hash and scheme refused; the recipe not checked. |
| Cross-host resume | One reviewer run, `c2-rev-verify-final-s51200-a1`: `c2train.py --resume` from the recovered r1 `rnn-final-s51200` `ckpt-1000` to 10,000 with `--step-weights 0 0 1`, on this host. Its metrics are compared with r1's published rows, and its `ckpt-10000` arrays with the recovered original. It doubles as my host calibration. | Identical on this host too. This is the third host for that run. |

## 2. Lane reviews

Each lane's QUESTION.md is read when its M0 lands. My counterexample for each is recorded under Revisions below before I open its result tables.

## Revisions

(Dated, before the matching analysis or run.)

- **04:06, c2probe added to the pin check** (b0fb22c; written after the first `rev_pin.py` pass, sections 0–7, and before any probe was fitted). `c2probe.decode_table` on the recovered `r1-comp-rnn-inter-s51200-a1` `ckpt-10000` at probe seed 64300 (D5). Checks: its labels equal the oracle's symbols at positions 0–3; each probe's weights equal r1's own `composition/analyze.py fit_probe` on the same activations, bitwise; every table entry, re-scored with my own forward and argmax, equals `table[i][j]` with i the fitted position and j the scored one (a transposed table fails this); the diagonal equals r1's held-out probe accuracy. *Prediction:* all identical.
