from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        
        for layer in self.net.layers:
            for x, v in layer.params.items():
                if x not in self.velocity.keys():
                    self.velocity[x] = np.zeros(layer.params[x].shape)
                self.velocity[x] = (self.momentum * self.velocity[x]) - (self.lr * layer.grads[x])
                layer.params[x] = (layer.params[x] + self.velocity[x])

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        
        #set all the caches to zero initially
        
        if not self.cache:
            for layer in self.net.layers:
                for x, y in list(layer.params.items()):
                    self.cache[x] = np.zeros_like(y)
        
        for layer in self.net.layers:
            for x, y in list(layer.params.items()):
                dv = layer.grads[x]
                self.cache[x] = self.decay * self.cache[x] + (1 - self.decay) * np.square(dv)
                layer.params[x] += - (self.lr * dv) / (np.sqrt(self.cache[x]) + self.eps)
                
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        
        for layer in self.net.layers:
            for n, v in list(layer.params.items()):
                
                dg = layer.grads[n]
                self.t += 1
                
                if n not in self.mt:
                    self.mt[n] = np.zeros_like(dg)
                if n not in self.vt:
                    self.vt[n] = np.zeros_like(dg)
                    
                #g_t = grad_func(theta_0)                    #computes the gradient of the stochastic function
                self.mt[n] = self.beta1 * self.mt[n] + (1 - self.beta1) * dg           #updates the moving averages of the gradient
                self.vt[n] = self.beta2 * self.vt[n] + (1 - self.beta2) * (dg * dg)  #updates the moving averages of the squared gradient
                m_cap = self.mt[n] / (1 - (self.beta1 ** self.t))                 #calculates the bias-corrected estimates
                v_cap = self.vt[n] / (1 - (self.beta2 ** self.t))                 #calculates the bias-corrected estimates
                layer.params[n] = layer.params[n] - ((self.lr * m_cap) / (np.sqrt(v_cap) + self.eps))     #updates the parameters
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
