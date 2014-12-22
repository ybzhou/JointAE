import theano

import theano.tensor as T
import numpy as np

from AELayer import AELayer, Gaussian_AELayer, Bernoulli_AELayer, \
                    SaltPepper_AELayer, Contractive_AELayer
                    

class BasicAE(object):
    def __init__(self, en_input, de_input, n_in, n_hid, init, act_func,
                 tie_weights=False, We=None, be=None, Wd=None, bd=None, **kargs):
        
        self.encoder = AELayer(n_in, n_hid, init, act_func[0], W=We, b=be, **kargs)
        self.tie_weights = tie_weights
        self.params = []
        self.params.extend(self.encoder.params)
        if tie_weights:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=self.encoder.W.T, b=bd, **kargs)
            self.params.append(self.decoder.b)
        else:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=Wd, b=bd, **kargs)
            self.params.extend(self.decoder.params)
            
    def cost(self):
        ''' L2 cost for ae layer'''
        if self.tie_weights:
            return self.encoder.layer_cost()
        else:
            return self.encoder.layer_cost() + self.decoder.layer_cost()
        
#-------------------------------------------------------------------------------
class GaussianDAE(object):
    def __init__(self, n_in, n_hid, init, act_func,
                 tie_weights=False, We=None, be=None, Wd=None, bd=None, **kargs):
        
        self.encoder = Gaussian_AELayer(n_in, n_hid, init, act_func[0], W=We, b=be, **kargs)
        self.params = []
        self.params.extend(self.encoder.params)
        self.tie_weights = tie_weights
        if tie_weights:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=self.encoder.W.T, b=bd, **kargs)
            self.params.append(self.decoder.b)
        else:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=Wd, b=bd, **kargs)
            self.params.extend(self.decoder.params)
            
    def cost(self):
        ''' L2 cost for ae layer'''
        if self.tie_weights:
            return self.encoder.layer_cost()
        else:
            return self.encoder.layer_cost() + self.decoder.layer_cost()
        
#-------------------------------------------------------------------------------
class BernoulliDAE(object):
    def __init__(self, n_in, n_hid, init, act_func,
                 tie_weights=False, We=None, be=None, Wd=None, bd=None, **kargs):
        self.encoder = Bernoulli_AELayer(n_in, n_hid, init, act_func[0], W=We, b=be, **kargs)
        self.tie_weights = tie_weights
        self.params = []
        self.params.extend(self.encoder.params)
        if tie_weights:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=self.encoder.W.T, b=bd, **kargs)
            self.params.append(self.decoder.b)
        else:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=Wd, b=bd, **kargs)
            self.params.extend(self.decoder.params)
            
    def cost(self):
        ''' L2 cost for ae layer'''
        if self.tie_weights:
            return self.encoder.layer_cost()
        else:
            return self.encoder.layer_cost() + self.decoder.layer_cost()
        
#-------------------------------------------------------------------------------
class SaltPepperDAE(object):
    def __init__(self, n_in, n_hid, init, act_func,
                 tie_weights=False, We=None, be=None, Wd=None, bd=None, **kargs):
        
        self.encoder = SaltPepper_AELayer(n_in, n_hid, init, act_func[0], W=We, b=be, **kargs)
        self.tie_weights = tie_weights
        self.params = []
        self.params.extend(self.encoder.params)
        if tie_weights:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=self.encoder.W.T, b=bd, **kargs)
            self.params.append(self.decoder.b)
        else:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=Wd, b=bd, **kargs)
            self.params.extend(self.decoder.params)
            
    def cost(self):
        ''' L2 cost for ae layer'''
        if self.tie_weights:
            return self.encoder.layer_cost()
        else:
            return self.encoder.layer_cost() + self.decoder.layer_cost()
        
#-------------------------------------------------------------------------------
class ContractiveAE(object):
    def __init__(self, n_in, n_hid, init, act_func,
                 tie_weights=False, We=None, be=None, Wd=None, bd=None, **kargs):
        
        self.encoder = Contractive_AELayer(n_in, n_hid, init, act_func[0], W=We, b=be, **kargs)
        self.tie_weights = tie_weights
        self.params = []
        self.params.extend(self.encoder.params)
        if tie_weights:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=self.encoder.W.T, b=bd, **kargs)
            self.params.append(self.decoder.b)
        else:
            self.decoder = AELayer(n_hid, n_in, init, act_func[1], W=Wd, b=bd, **kargs)
            self.params.extend(self.decoder.params)
            
    def cost(self):
        ''' L2 cost for ae layer'''
        if self.tie_weights:
            return self.encoder.layer_cost()
        else:
            return self.encoder.layer_cost() + self.decoder.layer_cost()