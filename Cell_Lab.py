# asymmetric interacting particles under active noise
# 2D periodic boundary condition
# Yunsik Choe, Seoul national University

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys



class Cell_Lab:     # OOP
    """basic model to simulate 2D passive objects under active noise"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,L, N_ptcl,Fs):
        
        
        # set up coefficients
        self.set_coeff(L,N_ptcl,Fs) 
      
        # initializing configuration of state
        self.set_zero()
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_ptcl,Fs):
        self.L=L
        self.N_ptcl=N_ptcl
        self.Fs=Fs
        self.dt = 1/Fs
        self.D = 1
        self.tau = 1
        
                  
        
        
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.etaX = np.random.normal(0,np.sqrt(self.D/self.tau),N_ptcl) 
        self.etaY = np.random.normal(0,np.sqrt(self.D/self.tau),N_ptcl) 
        
        self.X = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
        self.Y = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
    
    def noise_evolve(self):             # random part of s dynamics
        xiX = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau**2),N_ptcl) 
        xiY = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau**2),N_ptcl)
        self.etaX
    
    def dynamics(self,x,s):         # dynamics of x and s to give time difference of x and s
      
        
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
    
