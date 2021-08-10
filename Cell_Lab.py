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
        
        self.record = False
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_ptcl,Fs):
        
        # system coefficients
        self.L=L
        self.N_ptcl=N_ptcl
        self.Fs=Fs
        self.dt = 1/Fs
        
        self.N_skip = 50
        
        # noise coefficients
        self.D = 20
        self.Dr = 20
        self.tau_noise = 1
        
        # dynamics
        self.memory = False
        self.tau_momentum = 0.1
        self.p = 1
        
        # inner structure coefficients
        self.n = 3
        self.l = [-1,1,3]
        self.r = [3,2,1]
        self.k = [10,5,5]
        self.mu = 1
        self.mur = 0.2
        self.poten_order = 3
        
        self.initialize = False
        self.grid_ordered = False
        self.ang_ordered = False
        

                  
        
        
    # boundary condition
    
    def periodic(self,x,y):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        mod_y = -self.L/2   +    (y+self.L/2)%self.L               # returns -L/2 ~ L/2

        return (mod_x,mod_y)
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.etaX = np.random.normal(0,np.sqrt(self.D/self.tau_noise),self.N_ptcl) 
        self.etaY = np.random.normal(0,np.sqrt(self.D/self.tau_noise),self.N_ptcl) 
        self.etaO = np.random.normal(0,np.sqrt(self.Dr/self.tau_noise),self.N_ptcl)
        
        self.FX = 0
        self.FY = 0
        self.Torque = 0
        
        if self.grid_ordered:
            grid = np.linspace(0,self.L,int(np.ceil(np.sqrt(self.N_ptcl)))+1)
            xgrid,ygrid = np.meshgrid(grid[:-1],grid[:-1])
            xgrid = xgrid.reshape(-1)[:self.N_ptcl]
            ygrid = ygrid.reshape(-1)[:self.N_ptcl]

            self.X = xgrid
            self.Y = ygrid
         
            
        else: 
        
            self.X = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
            self.Y = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
            
            
        if self.ang_ordered=='anti-parallel':
            self.O = np.ones(self.N_ptcl)*np.pi/6 + (np.pi/2)*(-1)**np.arange(self.N_ptcl)
        elif self.ang_ordered=='parallel':
            self.O = np.ones(self.N_ptcl)*np.pi/6 
        else:
            self.O = np.random.uniform(0,2*np.pi,self.N_ptcl)
        
        self.set_structure()
        
    def set_structure(self):
        self.Xs = self.X.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.cos(self.O)
        self.Ys = self.Y.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.sin(self.O)
        
        (self.Xs,self.Ys) = self.periodic(self.Xs,self.Ys)
    
    def noise_evolve(self):             # random part of s dynamics
        xiX = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau_noise**2),self.N_ptcl) 
        xiY = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau_noise**2),self.N_ptcl)        
        xiO = np.random.normal(0,np.sqrt(2*self.Dr*self.dt/self.tau_noise**2),self.N_ptcl)

        self.etaX = (1-self.dt/self.tau_noise)*self.etaX+xiX
        self.etaY = (1-self.dt/self.tau_noise)*self.etaY+xiY
        self.etaO = (1-self.dt/self.tau_noise)*self.etaO+xiO
#         self.etaO = xiO
        
    def force(self,i,j):    # force and torque by x,y to X,Y with axis at angle O with length l, with force r, k
        relXx = (self.Xs[i].reshape(-1,1)-self.Xs[j].reshape(1,-1))
        relYy = (self.Ys[i].reshape(-1,1)-self.Ys[j].reshape(1,-1))
        (relXx,relYy) = self.periodic(relXx,relYy)
        length = np.sqrt(relXx**2+relYy**2)
        
        interact1 = (length<self.r[i])
        interact2 = (length<self.r[j])
        
        fx     = np.sum((self.k[i]*(self.r[i]-length)**(self.poten_order-1)*interact1 + self.k[j]*(self.r[j]-length)*interact2)*np.divide(relXx,length,out=np.zeros_like(relXx),where=length!=0), axis=1)
        
        fy     = np.sum((self.k[i]*(self.r[i]-length)**(self.poten_order-1)*interact1 + self.k[j]*(self.r[j]-length)*interact2)*np.divide(relYy,length,out=np.zeros_like(relYy),where=length!=0), axis=1)
        
        torque = -fx*self.l[i]*np.sin(self.O) + fy*self.l[i]*np.cos(self.O)      # force acted on the given particle, angle 0 increase in fx=0, fy=1
        return(fx,fy,torque)

    
    def time_evolve(self):
        
        # compute force & torque
        FX = np.zeros(self.N_ptcl)
        FY = np.zeros(self.N_ptcl)
        Torque = np.zeros(self.N_ptcl)
        
        
        for i in range(self.n):
            for j in range(self.n):
                (fx,fy,torque) = self.force(i,j)
                FX     += fx
                FY     += fy
                Torque += torque


        # compute noise
        self.noise_evolve()        
        
        


        # memory in force (momentum)
        self.FX     = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  FX
        self.FY     = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  FY
        self.Torque = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  Torque
        
        
        
        # update configuration


        
        
        self.X += self.mu*(self.FX+self.etaX+self.p*np.cos(self.O))*self.dt
        self.Y += self.mu*(self.FY+self.etaY+self.p*np.sin(self.O))*self.dt

#         self.X += self.mu*(FX+self.etaX+self.p*np.cos(self.O))*self.dt
#         self.Y += self.mu*(FY+self.etaY+self.p*np.sin(self.O))*self.dt
        
    
        self.O += self.mur*(self.Torque+self.etaO)*self.dt
#         self.O += self.mur*(Torque+self.etaO)*self.dt
        
        (self.X,self.Y) = self.periodic(self.X,self.Y)
        self.set_structure()
        
    def initializing(self,N):
        L = self.L
        self.L = 3*L
        self.set_zero()
        print('initializing...')
        for i in trange(N):
            self.L = L*(1+9*(N-i)/N)
            self.time_evolve()
        self.L = L
        
        
    def animate(self,N_iter,directory):
        if self.initialize:
            self.initializing(2000)
        
        axrange = [-self.L/2, self.L/2, -self.L/2, self.L/2]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        for i in range(self.n):
            ax2.scatter(0,self.l[i],s=self.r[i]*500000/self.L**2)
        ax2.quiver(0,self.l[0],0,(self.l[-1]-self.l[0]),scale =self.L)
        ax2.axis(axrange)
        ax2.set_aspect('equal','box')
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        
        os.makedirs('record/'+str(directory),exist_ok=True)


        
        
        for nn in trange(N_iter):
            
            if self.record:
                ax1.clear()
                ax1.quiver(self.Xs[0],self.Ys[0],(self.l[-1]-self.l[0])*np.cos(self.O),(self.l[-1]-self.l[0])*np.sin(self.O),scale = self.L)
    #             for i in range(self.n):
    #                 ax1.scatter(self.Xs[i],self.Ys[i],s=self.r[i]*500000/self.L**2)
                ax1.axis(axrange)
                ax1.set_aspect('equal', 'box')
                fig1.canvas.draw()
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
    
    
       

