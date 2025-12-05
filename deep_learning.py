import numpy as np
import math
import random

"""
Copyright (c) 2025 Nils Häußler. All Rights Reserved.

This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
"""

np.set_printoptions(suppress=True)


class Utilty:
    """
    Copyright (c) 2025 Nils Häußler. All Rights Reserved.

    This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
    """
    
    def __init__(self):
        pass
    def get_loss_func(self,name):
        if name == "squared" or name == "mse":
            return self.mean_squared_error
        elif name == "linear":
            return self.linear_error
        
        raise Exception("No loss with the name {}".format(name))
    
    def mean_squared_error(self,result,desired):
        return 1/2*np.power(np.abs(desired-result),2)
    
    def linear_error(self,result,desired):
        return (desired-result)
        
    def get_loss_func_d(self,name):
        if name == "squared" or name == "mse":
            return self.mean_squared_error_d
        elif name == "linear":
            return self.linear_error_d
        
        raise Exception("No derived loss with the name {}".format(name))
    
    def mean_squared_error_d(self,result,desired):
        return (result-desired)
    
    def linear_error_d(self,result,desired):
        raise Exception("No derivative for linear error")
        return 0
    
    

class Perceptron(Utilty):
    """
    Copyright (c) 2025 Nils Häußler. All Rights Reserved.

    This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
    """
    
    def __init__(self,num_weights=3,aktivation_fun="sigmoid",l_rate=0.01, loss_func = "mse", dropout=0, temperature = 0.1):
        self.weights = np.random.random_sample((num_weights,)) - 0.5
        self.f_akt_desc = aktivation_fun
        
        self.dropout = dropout
        
        self.f_akt = self.get_f_akt()
        self.f_akt_derive = self.get_f_akt_derive()
        
        self.loss_func = self.get_loss_func(loss_func)
        self.loss_func_derive = self.get_loss_func_d(loss_func)
        
        self.temperature = temperature
        
        self.inv_temperature = 1/self.temperature
        
        
        
        self.rate = l_rate;
        
        self.clear_errors()
        self.form = num_weights
        self.leaky_relu_m = 1/10
    
    def clear_errors(self):
        self.current_error = None
        self.saved_value = None
        self.saved_net = None
        self.saved_delta = None
        self.dropped_out = False
        
    def check_input(self, net):
        if self.weights.shape[0] == net.shape[0]+1:
            net = np.append(net,1)
        
        if self.weights.shape[0] != net.shape[0]:
            print("shape misfit. weights:"+str(self.weights.shape[0])+", input:"+str(net.shape[0]))
            print("one is due to bias")
        return net
    
    def to_np(self, array):
        if isinstance(array, np.ndarray):
            return array
        return np.array(array)
        
    def net_sum(self, inputs):
        inputs = self.check_input(self.to_np(inputs))
        
        return np.dot(inputs, self.weights)
    
    def get_f_akt(self):
        if self.f_akt_desc == "sigmoid":
            return self.sigmoid
        
        if self.f_akt_desc == "heaviside":
            return self.heaviside
        
        if self.f_akt_desc == "relu":
            return self.relu
        
        if self.f_akt_desc == "leaky_relu":
            return self.leaky_relu
        
        raise Exception("no activation function under the name '{}'".format(self.f_akt_desc))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-self.inv_temperature*x))
    
    def heaviside(self, x):
        if x < 0:
            return 0
        if x==0:
            return 1/2
        if x > 0:
            return 1
    
    def relu(self, x):
        return max(0,x)
    
    def leaky_relu(self, x):
        return max(0,x) + min(x*self.leaky_relu_m,0)
    
    
    def get_f_akt_derive(self):
        if self.f_akt_desc == "sigmoid":
            return self.sigmoid_d
        
        if self.f_akt_desc == "heaviside":
            return self.heaviside_d
        
        if self.f_akt_desc == "relu":
            return self.relu_d
        
        if self.f_akt_desc == "leaky_relu":
            return self.leaky_relu_d
        
        raise Exception("no derivative of a function under the name '{}'".format(self.f_akt_desc))
    
    def sigmoid_d(self, x):
        s = self.sigmoid(x)
        return self.inv_temperature * (1-s) * s
    
    def heaviside_d(self, x):
        print("Warning!: There is no derivative for the heaviside function")
        return 0
    
    def relu_d(self, x):
        if x < 0:
            return 0
        return 1
    
    def leaky_relu_d(self, x):
        if x < 0:
            return self.leaky_relu_m
        return 1
    
    def feed_forward(self, inputs, dropout=True):
        
        self.saved_net = self.net_sum(inputs)
        
        self.saved_value = self.f_akt(self.saved_net)
        
        if random.random() < self.dropout and self.dropout != 0 and dropout:
            self.dropped_out = True
            return self.saved_value * 0
        
        return self.saved_value
    
    def train_simple(self,inputs,should):
        should = self.to_np(should)
        
        inputs = self.check_input(self.to_np(inputs))
        
        result = self.feed_forward(inputs)
        
        error = -self.loss_func(should,result)
        
        self.weights += inputs * error * self.rate
        
        return error
    
    def __str__(self):              
        return "---Perceptron---\n\t weights: {}\n\t learning rate: {}\n---------------".format(self.weights,self.rate)
    
    def calculate_delta(self,parent_layer,result,desired,input_,perceptron_index):
        if self.dropped_out:
            return 0
                
        if parent_layer.type == "output":
            return self.f_akt_derive(self.saved_net) * self.loss_func_derive(result,desired)[perceptron_index]
        else:
            following_delta_sum = 0
            
            for i in range(0,len(parent_layer.following_layer.perceptrons)):
                p = parent_layer.following_layer.perceptrons[i]
                
                if p.dropped_out:
                    continue
                
                following_delta_sum += p.saved_delta * p.weights[perceptron_index]
            
            return self.f_akt_derive(self.saved_net) * following_delta_sum
        
    def backpropagation(self, parent_layer,result,desired,input_,perceptron_index):
        delta_weights = []
        
        input_ = self.to_np(input_)
        
        self.saved_delta = self.calculate_delta(parent_layer,result,desired,input_,perceptron_index)
        
        for i in range(0,self.form):
            if i == self.form-1:
                preceding_output_value = 1
            elif parent_layer.preceding_layer == None:
                preceding_output_value = input_[i]
            else:
                preceding_output_value = parent_layer.preceding_layer.perceptrons[i].saved_value
            
            delta_w = -self.rate * self.saved_delta * preceding_output_value
            
            delta_weights.append(delta_w)
        
        self.weights += np.array(delta_weights)

class Layer:
    """
    Copyright (c) 2025 Nils Häußler. All Rights Reserved.

    This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
    """
    def __init__(self, layer_type="hidden",num_neurons=2, stats = None):
        self.stats = stats
        self.num_neurons = num_neurons
        self.perceptrons = []
        self.form = []
        
        self.following_layer = None
        self.preceding_layer = None
            
        self.type = layer_type
        assert self.type in ["hidden","output"], "layer_type must be one of 'input', 'hidden', 'output'"
        
    def clear_errors(self):
        for p in self.perceptrons:
            p.clear_errors()
    
    def build_from_layer(self, last_layer):
        self.stats["num_weights"] = last_layer.num_neurons+1 #plus bias
        
        for i in range(0,self.num_neurons):
            self.perceptrons.append(Perceptron(**self.stats))
        
        self.form = []
        for p in self.perceptrons:
            self.form.append(p.form)
    
    def build_from_input(self, input_size):
        self.stats["num_weights"] = input_size+1 #plus bias
        
        for i in range(0,self.num_neurons):
            self.perceptrons.append(Perceptron(**self.stats))
        
        self.form = []
        for p in self.perceptrons:
            self.form.append(p.form)
        
    def dense_input(self, inputs, dropout=True):
        output = [];
        
        for p in self.perceptrons:
            output.append(p.feed_forward(inputs, dropout))
        
        return np.array(output)
    
    def backpropagation(self, result,desired, input_):
        if self.type == "hidden":
            assert self.following_layer != None, "A hidden layer needs a following 'output' or 'hidden' layer. It cant be the output layer."
        
        for index,p in enumerate(self.perceptrons):
            p.backpropagation(self,result,desired,input_,index)

class Model(Utilty):
    """
    Copyright (c) 2025 Nils Häußler. All Rights Reserved.

    This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
    """
    
    def __init__(self, model_type="simple", input_size=2, loss_func = "mse", random_seed = None, neuron_stats = {}):
        np.random.seed(random_seed)
        self.n_stats = neuron_stats
            
        self.input_size = input_size
        self.layers = []
        self.form = []
        self.type = model_type
        self.loss_func = self.get_loss_func(loss_func)
    
    def add_layer(self, layer):
        if layer.stats == None:
            layer.stats = self.n_stats
        
        if len(self.layers) == 0:
            layer.build_from_input(self.input_size)
        else:
            layer.build_from_layer(self.layers[-1])
            self.layers[-1].following_layer = layer
            layer.preceding_layer = self.layers[-1]
        
        self.layers.append(layer)
        
        self.form.append({str(layer.type):layer.form})
    
    def __str__(self):
        description = "Model of type '{}'".format(self.type)
        description += "\n--------------------\n"
        description += "Includes {} Layers.\n".format(len(self.layers))
        description += "Input dimension is {}\n".format(self.input_size)
        description += "Output dimension is {}\n".format(self.layers[-1].num_neurons)
        description += "--------------------\n"
        description += "The {} layers are:\n".format(len(self.layers))
        
        for l in self.layers:
            description += "\tLayer of type '{}', with {} neurons, with {} weights each\n".format(l.type, l.num_neurons, l.perceptrons[0].form)
        
        description += "--------------------"
        return description
    
    def feed_forward(self, input_, dropout=True):
        buffer = input_
        
        for l in self.layers:
            buffer = l.dense_input(buffer, dropout)
        
        return buffer
    def clear_errors(self):
        for l in self.layers:
            l.clear_errors()
    
    def train(self,input_,desired):
        self.clear_errors()
        
        result = self.feed_forward(input_)
        
        for layer in reversed(self.layers):
            layer.backpropagation(result,desired,input_)
        
        return self.eval(input_,desired)
    
    def eval(self, input_, desired):
        self.clear_errors()
        
        result = self.feed_forward(input_,False)
        
        return (np.abs(desired-result).sum()**2)