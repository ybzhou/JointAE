import theano
import gzip
import cPickle
import numpy
import time

import theano.tensor as T

from JointAE import JointAE
from Cost import CrossEntropyCost
from WeightInit import UniformWeightInit
from Optimizer import SGD_Optimizer, SGD_Momentum_Optimizer
from AE import BernoulliDAE, GaussianDAE



def test_mnist():
    seed=123
    batch_size = 100
    n_epochs = 10
    f = gzip.open('/data/Research/datasets/mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    print train_set[0].shape,train_set[1].shape
    f.close()
    
    n_samples = train_set[0].shape[0]
    train_set_x = theano.shared(
                numpy.asarray(train_set[0],dtype=theano.config.floatX),
                borrow=True)

    ae = JointAE(num_vis = 28*28, 
                 num_hiddens = [1000,500], 
                 layer_types = [BernoulliDAE, BernoulliDAE], 
                 autoencoder_act_funcs = [T.nnet.sigmoid, T.nnet.sigmoid],
                 output_act_func = T.nnet.sigmoid,
                 initializer = UniformWeightInit(seed),
                 cost_class = CrossEntropyCost,
                 tie_weights = False, 
                 init_params = None, 
                 noise=0.25, reg=0.)
    
    cost = ae.get_cost()
#     opt = SGD_Optimizer({'learning_rate':0.1})
    opt = SGD_Momentum_Optimizer({'learning_rate':0.1, 'momentum':0.9})
    updates = opt.get_updates(cost, ae.params)
    
    index = T.lscalar('index')
    train_func = theano.function(inputs=[index],
              outputs=cost,
              updates=updates,
              givens={
                ae.x: train_set_x[index*batch_size : (index+1)*batch_size],
                      },
              name='train',allow_input_downcast=True,on_unused_input='warn')
    
    
    
    
    n_batches = n_samples/batch_size
    
        
    epoch = 0
    train_e = numpy.zeros(n_batches)
    while (epoch < n_epochs):
        epoch = epoch + 1
        start_time = time.clock()
        
        for minibatch_index in xrange(n_batches):
            train_e[minibatch_index] = train_func(minibatch_index)
            
        train_error = numpy.mean(train_e)
        end_time = time.clock()
        print "epoch %d, training error %f, elapsed %.4fs"  % \
           (epoch, train_error, end_time-start_time)
                

def test(b=1,c=2, **kargs):
    a = kargs
    print a
if __name__ == '__main__':
    test_mnist()
#     test(d=1)
