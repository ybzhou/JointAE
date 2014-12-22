import abc

import theano.tensor as T

class Cost(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, input, target):
        self.x = input
        self.t = target
        
    @abc.abstractmethod
    def get_cost(self):
        ''' return the objective cost '''
        return

    
''' Binary cross entropy cost'''
class CrossEntropyCost(Cost):
    
    def get_cost(self):
        L = - T.sum(self.t * T.log(self.x) + (1 - self.t) * T.log(1 - self.x), axis=1)
        return T.mean(L)

''' Sum of square cost '''
class SSqCost(Cost):
    
    def get_cost(self):
        L = T.sum(T.sqr(self.t-self.x), axis=1)
        return T.mean(L)