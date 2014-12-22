import abc
import theano

import numpy as np

class WeightInit:
    def __init__(self, seed, args=None):
        self._seed = seed
        self._args = args
        self._rng = np.random.RandomState(seed)
    
    @abc.abstractmethod
    def init(self, num_vis, num_hid):
        raise('Not specified error')
        pass

class SparseWeightInit(WeightInit):
    def init(self, num_vis, num_hid):
        num_init = self._args['num_init']
        std = self._args['std']
        W = np.asarray(self._rng.normal(size=(num_vis, num_hid),
                      scale=self.std), dtype=theano.config.floatX)
        for i in xrange(num_hid):
            perm_idx = self._rng.permutation(num_vis)
            W[perm_idx[:self._nv-num_init+1],i] = 0
        
        return W

class UniformWeightInit(WeightInit):
    def init(self, num_vis, num_hid):
        W = np.asarray(self._rng.uniform(
            low=-4*np.sqrt(6. / (num_vis + num_hid)),
            high=4*np.sqrt(6. / (num_vis + num_hid)),
            size=(num_vis, num_hid)), dtype=theano.config.floatX)
        
        return W
    
class GaussianWeightInit(WeightInit):
    def init(self, num_vis, num_hid):
        std = self._args['std']
        W = np.asarray( 
            self._rng.normal(size=(num_vis, num_hid), scale=std), 
            dtype=theano.config.floatX)
        
        return W
    
class ConstantInit(WeightInit):
    def init(self, num_vis, num_hid):
        c = self._args['std']
        W = np.asarray(
            c*np.ones(size=(num_vis, num_hid)), dtype=theano.config.floatX)
        return W