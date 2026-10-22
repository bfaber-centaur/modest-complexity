# c2 mechanisms: adaptive choices

Choices made after QUESTION.md was committed, in order, with the reason and what had been seen when each was made. Nothing is removed from this log.

| time (UTC) | choice | reason | seen before choosing |
|---|---|---|---|
| 04:09 | Calibration ran the full interchange script on the nine `ckpt-0` files (`c2-mech-calib-a1`), not a separate timing stub | It times the real job; `ckpt-0` results are labelled calibration and are duplicated in `c2-mech-interchange-a1` | nothing beyond QUESTION.md |
| 04:12 | Primary figure as planned. The right column draws S_cent as one marker per cell at its L, not a line, and adds a supporting `figures/centroid.png` (S_cent matrices beside natural S) | S_cent is not constant along diagonals, so a per-L line would average distinct cells. The centroid result turned out to be the lane's main non-equivalent observation | M2 numbers (`cells.csv`) |
| 04:12 | `both_correct` and held-out-history subsets stay in `pairs.csv` and the README, not a second panel row | they differ visibly only where the subset definition forces it ((3, 3) = 1 by construction) | `pairs.csv` |
