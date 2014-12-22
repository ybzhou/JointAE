import theano.tensor as T

class JointAE(object):
    def __init__(self, num_vis, num_hiddens, layer_types, autoencoder_act_funcs,
                 output_act_func, initializer,
                 cost_class, tie_weights,
                 init_params, **kargs):
        
        self.hidden_layers = []
        self.params = []
        self.num_hidden_layers = len(num_hiddens)
        self.x = T.matrix('x')
        self.cost_func = cost_class
        self.tie_weights = tie_weights
        
        param_idx = 0
        for layer_idx in xrange(self.num_hidden_layers):
            if init_params is not None:
                if tie_weights:
                    We = init_params[param_idx]
                    be = init_params[param_idx+1]
                    bd = init_params[param_idx+2]
                    param_idx += 3
                else:
                    We = init_params[param_idx]
                    Wd = init_params[param_idx+1]
                    be = init_params[param_idx+2]
                    bd = init_params[param_idx+3]
                    param_idx += 4
            else:
                We = None
                Wd = None
                be = None
                bd = None
                
            if layer_idx == 0:
                layer_input = self.x
                num_input = num_vis
                num_output = num_hiddens[layer_idx]
                act_funcs = [autoencoder_act_funcs[layer_idx], 
                             output_act_func]
            else:
                layer_input = self.hidden_layers[-1].encoder.output
                num_input = num_hiddens[layer_idx-1]
                num_output = num_hiddens[layer_idx]
                act_funcs = [autoencoder_act_funcs[layer_idx], 
                             autoencoder_act_funcs[layer_idx-1]]
                
            layer = layer_types[layer_idx](
                        #input = layer_input, 
                        n_in = num_input, 
                        n_hid = num_output,
                        init = initializer,
                        act_func = act_funcs,
                        tie_weights = tie_weights,
                        We = We,
                        Wd = Wd,
                        be = be,
                        bd = bd,
                        **kargs)
            
            self.hidden_layers.append(layer)
            self.params.extend(layer.params)
            
         
    
    def get_reconstruction(self, x):
        num_layers = len(self.hidden_layers)
        input = x
        for layer_idx in xrange(num_layers):
            output = self.hidden_layers[layer_idx].encoder.fprop(input)
            input = output
            
        rec = output
        for layer_idx in xrange(num_layers-1, -1, -1):
            output = self.hidden_layers[layer_idx].decoder.fprop(rec)
            rec = output
        return rec
    
    def get_cost(self):

        reconstructed = self.get_reconstruction(self.x)

        C = self.cost_func(reconstructed, self.x)
        
        rec_cost = C.get_cost()
        
#         ae_cost = T.scalar('ae_cost')
#         for i in xrange(len(self.hidden_layers)):
#             ae_cost += self.hidden_layers[i].cost()
        
        return rec_cost #+ ae_cost
