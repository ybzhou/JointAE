import theano

import theano.tensor as T
import numpy as np

class AELayer(object):
    
    def __init__(self, n_in, n_out, init, act_func,
                 W=None, b=None, **kargs):
        self.x = T.matrix('x')
        self.act_func = act_func
        self.args = kargs

        self.theano_rng = T.shared_randomstreams.RandomStreams(init._seed)
        
        if W is None:
            W = theano.shared(value=init.init(n_in, n_out), name='W', borrow=True)
        self.W = W
        
        if b is None:
            b = theano.shared(value=-np.ones(n_out, dtype=theano.config.floatX),
                              name='b', borrow=True)
        self.b = b
        
        self.pre_act = T.dot(self.x, self.W) + self.b
        self.output = self.act_func(self.pre_act)
        self.params = [self.W, self.b]
    
    def fprop(self, input):
        pre_act = T.dot(input, self.W) + self.b
        return self.act_func(pre_act)
    
    def layer_cost(self):
        ''' L2 cost for ae layer'''
        return T.cast(self.args['reg']*T.sum(T.sqr(self.W)), theano.config.floatX)

################################################################################
class Gaussian_AELayer(AELayer):
    def fprop(self, input):
        x_bar = input + self.theano_rng.normal(size=input.shape, avg=0.0, 
                                          std=self.args['noise'], 
                                          dtype=theano.config.floatX)
        pre_act = T.dot(x_bar, self.W) + self.b
        return T.cast(self.act_func(pre_act), theano.config.floatX)

################################################################################
class Bernoulli_AELayer(AELayer):
    def fprop(self, input):
        x_bar = input * self.theano_rng.binomial(size=input.shape, n=1, 
                                          p=1 - self.args['noise'], 
                                          dtype=theano.config.floatX)
        pre_act = T.dot(x_bar, self.W) + self.b
        return T.cast(self.act_func(pre_act), theano.config.floatX)
    
################################################################################
class SaltPepper_AELayer(AELayer):
    def fprop(self, input):
        mask = self.theano_rng.binomial(size=input.shape, n=1, p=1 - self.args['noise'],
                                             dtype=theano.config.floatX)
        noise = self.theano_rng.binomial(size=input.shape, n=1, p=0.5,
                                             dtype=theano.config.floatX)
        x_bar = mask*input + T.eq(mask,0)*noise
        
        pre_act = T.dot(x_bar, self.W) + self.b
        return T.cast(self.act_func(pre_act), theano.config.floatX)
    
################################################################################
class Contractive_AELayer(AELayer):
    def layer_cost(self, input):
        ''' contractive cost'''
        pre_act = T.dot(input, self.W) + self.b
        h = self.act_func(pre_act)
        act_grad = T.grad(h.sum(), pre_act)
        frob_norm = T.dot(T.sqr(act_grad), T.sqr(self.W).sum(axis=0))
        contract_pen = frob_norm.sum() / input.shape[0]
        
        ''' L2 cost '''
        l2_cost = self.args['reg']*T.sum(T.sqr(self.W))
        return T.cast(self.args['contraction']*contract_pen+self.args['reg']*l2_cost, theano.config.floatX)