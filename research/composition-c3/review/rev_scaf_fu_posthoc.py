"""Post-hoc (after reading the follow-up NOTE): the reset effect on held-out length-3 and notN accuracy,
reset minus carried at equal logged steps 1,100-9,000, in keep (A'-A) and drop (B'-B), with A's own value
to show where keep nears its ceiling. Reads logged rows only (the lane's runs as extracted by rev_scaf_fu.py).

python3 research/composition-c3/review/rev_scaf_fu_posthoc.py runs/c3/c3-rev-scaf-fu-a1
"""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from rev_c3pin import R1, rows  # noqa: E402
from rev_scaf import subsets  # noqa: E402

SEEDS = (51200, 51201, 51202)


def main(out):
    L = Path(out)/'lane'/'research'/'composition-c3'/'scaffolding'/'runs'
    by = lambda path: {r['step']: r for r in rows(path)}
    A = {s: by(R1/'composition'/'runs'/f'r1-comp-rnn-inter-s{s}-a1'/'metrics.jsonl') for s in SEEDS}
    Ap = {s: by(L/f'c3-scaf-keep-reset-at1k-s{s}-a1'/'metrics.jsonl') for s in SEEDS}
    B = {s: by(L/f'c3-scaf-drop-at1k-s{s}-a1'/'metrics.jsonl') for s in SEEDS}
    Bp = {s: by(L/f'c3-scaf-drop-reset-at1k-s{s}-a1'/'metrics.jsonl') for s in SEEDS}
    h = lambda X, s, t: X[s][t]['held']['final']['acc']
    n = lambda X, s, t: subsets(X[s][t]['held_final_pred'])['notN']
    f = lambda xs: '/'.join(f'{x:+.3f}' for x in xs)
    print('step | held L3: A (keep) | A\'-A | B\'-B | notN: A\'-A | B\'-B')
    for t in (1100, 1250, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000):
        print(t, '|', '/'.join(f'{h(A, s, t):.3f}' for s in SEEDS), '|', f([h(Ap, s, t) - h(A, s, t) for s in SEEDS]), '|',
              f([h(Bp, s, t) - h(B, s, t) for s in SEEDS]), '|', f([n(Ap, s, t) - n(A, s, t) for s in SEEDS]), '|', f([n(Bp, s, t) - n(B, s, t) for s in SEEDS]))


if __name__ == '__main__':
    main(sys.argv[1])
