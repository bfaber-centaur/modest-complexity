import unittest
import numpy as np
from experiment import adam, init, lossgrad, objective_metrics


class InstrumentationTests(unittest.TestCase):
    def test_updates_match_original_and_decompose(self):
        w=init(7,9,12); old=[x.copy() for x in w]
        a,b=np.indices((7,7)).reshape(2,-1)
        _,g=lossgrad(w,a,b,(a+b)%7)
        m=[np.zeros_like(x) for x in w]; v=[x.copy() for x in m]
        stats=adam(w,g,m,v,1,.001,1.,measure=True)
        adaptive=[-.001*x/(np.sqrt(x*x)+1e-8) for x in g]
        decay=[-.001*x for x in old]
        for actual,x,ga in zip(w,old,g):
            expected=x*(1-.001)-.001*(.1*ga/(1-.9))/(np.sqrt(.001*ga**2/(1-.999))+1e-8)
            np.testing.assert_array_equal(actual,expected)
        self.assertAlmostEqual(stats['adaptive_norm'],np.sqrt(sum(np.sum(x*x) for x in adaptive)))
        self.assertAlmostEqual(stats['decay_norm'],np.sqrt(sum(np.sum(x*x) for x in decay)))
        self.assertAlmostEqual(stats['cosine'],sum(np.sum(x*y) for x,y in zip(adaptive,decay))/
                               (stats['adaptive_norm']*stats['decay_norm']))
        self.assertIsNone(adam(w,g,m,v,2,.001,0,measure=True)['cosine'])

    def test_loss_shift_invariance_and_margin(self):
        x=np.array([[2.,0.,1.],[0.,3.,1.]])
        y=np.array([0,2]); ix=np.arange(2)
        z=objective_metrics(x,y,ix)
        other=objective_metrics(x+1000,y,ix)
        for k in z:self.assertAlmostEqual(z[k],other[k])
        self.assertEqual(z['margin_min'],-2.)
        self.assertEqual(z['margin_median'],-.5)
        expected=np.mean(np.log(np.exp(x).sum(1))-x[ix,y])
        self.assertAlmostEqual(z['cross_entropy'],expected)

if __name__=='__main__':unittest.main()
