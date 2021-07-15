# asymmetric interacting particles under active noise
# Yunsik Choe, Seoul national University

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys



class Cell_Lab:     # OOP
    """basic model to simulate passive objects under active noise"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self):
        
        
        # set up coefficients
        self.set_coeff(alpha, u, len_time, N_time, N_X,N_ptcl,v,mu, muw) 
      
        # initializing configuration of state
        self.set_zero()
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self):
        
        
        
        
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.time = np.linspace(0,self.len_time,num=self.N_simul)
        self.s = np.random.choice([1,-1],np.array([self.N_ptcl,self.N_X]))             # random direction at initial time
        self.x = np.random.uniform(-self.L/2, self.L/2, size=np.array([self.N_ptcl,self.N_X]))     # starting with uniformly distributed particles
        self.X = np.zeros(self.N_X)
#         self.v = np.zeros(self.N_X)
    
    def tumble(self):             # random part of s dynamics
        tumble = np.random.choice([1,-1], size=np.array([self.N_ptcl,self.N_X]), p = [1-self.delta_time*self.alpha/2, self.delta_time*self.alpha/2]) # +1 no tumble, -1 tumble
        return tumble
    
    def dynamics(self,x,s):         # dynamics of x and s to give time difference of x and s
        dxa = self.u*self.delta_time
        ds = self.tumble()
        
        dx = dxa*s  -self.mu*self.partial_V(x)*self.delta_time
        if self.model==3:
            v = self.muw*np.sum(self.partial_V(x),axis=0)
        elif self.model ==2:
            v = self.v


        return (v, dx, ds)
        
    def time_evolve(self):
        (v, dx, ds) = self.dynamics(self.x, self.s)
        
        self.v = v
        if self.compute:
            self.dS1 = -np.average(self.partial_V(self.x+dx/2-v*self.delta_time/2)*(dx-v*self.delta_time),axis=0)
            self.dS2 = (self.u/self.mu)*np.average(self.s*dx,axis=0)
        
          
        self.x += dx                     # active particles movement
        self.s *= ds                     # direction of movement changed if the tumble is true
        self.X += v*self.delta_time           # passive object movement
        
        self.x = self.periodic(self.x)
        self.X = self.periodic(self.X)
        
            
            
  
            # computation part with time evolving
        for _ in trange(self.N_time*duration):
            self.time_evolve()
            F_v += (self.L/self.N_ptcl) *np.sum(self.partial_V(self.x),axis=0)/(self.N_time*duration)              # summing the V' at each x
            dS1_v += self.dS1/(self.N_time*duration)
            dS2_v += self.dS2/(self.N_time*duration)
            
        plt.plot(self.v/self.u,F_v)
        plt.xlabel('v/u')
        plt.ylabel('Force_wall')
        plt.show()
        plt.plot(self.v/self.u,dS1_v)
        plt.xlabel('v/u')
        plt.ylabel('dS1')
        plt.show()
        plt.plot(self.v/self.u,dS2_v)
        plt.xlabel('v/u')
        plt.ylabel('dS2')
        plt.show()
        plt.plot(self.v/self.u,dS1_v+dS2_v,'r')
        plt.xlabel('v/u')
        plt.ylabel('dSt')
        plt.show()
        
        return ((self.v/self.u,F_v,dS1_v,dS2_v))
    
