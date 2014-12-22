import abc
import theano

import theano.tensor as T
import numpy as np

class Optimizer(object):
    
    def __init__(self, kargs):
        self.args = kargs
    
    @abc.abstractmethod
    def get_updates(self, cost, theta):
        return
    
class SGD_Optimizer(Optimizer):
    
    def get_updates(self, cost, theta):
        
        lr = self.args['learning_rate']
        # get gradient
        gradients = T.grad(cost, theta)
        updates = []
        for (param, gparam) in zip(theta, gradients):
            updates.append((param, param - lr*gparam))
        
        return updates
    
class SGD_Momentum_Optimizer(Optimizer):
    def get_updates(self, cost, theta):
        self.momentum = []
        
        for i in xrange(len(theta)):
            val = np.zeros((theta[i].get_value().shape), dtype=theano.config.floatX)
            self.momentum.append(theano.shared(value=val, borrow=True))
        
        lr = self.args['learning_rate']
        mu = self.args['momentum']
        
        gradients = T.grad(cost, theta)
        updates = []
        for (param, gparam, mom) in zip(theta, gradients, self.momentum):
            delta = mu*mom - lr*gparam
            updates.append((mom, delta))
            updates.append((param, param + delta))
        
        return updates
    