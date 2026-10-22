"""Checks of continuation.py's measures on the start files (no continuation run needed).

uv run --locked --group analysis python -m unittest discover -s research/e3/mechanisms -v
"""
import csv
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuation as c  # noqa: E402

UPDATE0 = {r['run_id']: r for r in csv.DictReader(open(c.HERE/'starts/update0.csv'))}


def load(run_id):
    r = UPDATE0[run_id]
    with np.load(c.ROOT/r['start_file']) as z:
        ws = [z[k] for k in 'UVW']
    src = c.ROOT/'research'/('coordinator' if '-co-' in r['source_run'] else 'size-time')/'runs'/r['source_run']
    with np.load(src/'checkpoint_050000.npz') as z:
        wsrc = [z[k] for k in 'UVW']
    d = dict(np.load(src/'data.npz'))
    return ws, wsrc, d


class Continuation(unittest.TestCase):
    def test_update0_values_match_the_edit_report(self):
        for run_id, r in UPDATE0.items():
            ws, wsrc, d = load(run_id)
            sets = c.ladder_sets(d['a'], d['b'], c.tr30, c.tr50, c.COMMON, c.P)
            m = c.common_measures(ws, d, sets)
            assert abs(m['common_acc'] - float(r['common_acc'])) < 1e-12
            assert abs(c.own_heldout(ws, d) - float(r['own_heldout_acc'])) < 1e-12
            assert c.distance(ws, ws) == 0
            assert m['P_SN_undefined'] == m['P_cN_undefined'] == m['P_cS_undefined'] == 0

    def test_frequency_structure_of_the_three_arms(self):
        for run_id, r in UPDATE0.items():
            ws, wsrc, d = load(run_id)
            s = c.structure(ws, c.dominant(wsrc)[0], c.dominant(ws)[0])
            assert s['dominant_as_at_update0'] == 1
            if r['arm'] == 'filt':  # all non-DC input energy at the source dominant frequency
                assert abs(s['share_at_source_dominant'] - 1) < 1e-9 and abs(s['input_concentration'] - 1) < 1e-9
            elif r['arm'] == 'sham':  # none of it
                assert s['share_at_source_dominant'] < 1e-20
            else:
                assert 0 < s['share_at_source_dominant'] < 1

    def test_scale_free_measures_ignore_positive_output_scaling(self):
        ws, _, d = load('e3-me-filt-w128-i30001-a1')
        sets = c.ladder_sets(d['a'], d['b'], c.tr30, c.tr50, c.COMMON, c.P)
        m, m2 = c.common_measures(ws, d, sets), c.common_measures([ws[0], ws[1], 7.5*ws[2]], d, sets)
        for k in ('P_SN', 'P_cN', 'P_cS', 'common_acc', 'margin_N_sigma', 'gap_SN_sigma'):
            assert abs(m[k] - m2[k]) < 1e-12, k


if __name__ == '__main__':
    unittest.main()
