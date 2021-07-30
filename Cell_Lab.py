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
        self.tau = 1
        self.p = 1
        
        # inner structure coefficients
        self.l1 = 1
        self.l2 = 2
        self.r1 = 3
        self.r2 = 2
        self.k1 = 10
        self.k2 = 5
        self.mu = 1
        self.mur = 0.2
        

                  
        
        
    # boundary condition
    
    def periodic(self,x,y):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        mod_y = -self.L/2   +    (y+self.L/2)%self.L               # returns -L/2 ~ L/2

        return (mod_x,mod_y)
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.etaX = np.random.normal(0,np.sqrt(self.D/self.tau),self.N_ptcl) 
        self.etaY = np.random.normal(0,np.sqrt(self.D/self.tau),self.N_ptcl) 
        self.etaO = np.random.normal(0,np.sqrt(self.D/self.tau),self.N_ptcl) 
        
        self.X = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
        self.Y = np.random.uniform(-self.L/2,self.L/2,self.N_ptcl)
        self.O = np.random.uniform(0,2*np.pi,self.N_ptcl)
        
        self.set_structure()
        
    def set_structure(self):
        self.X1 = self.X - self.l1*np.cos(self.O)
        self.X2 = self.X + self.l2*np.cos(self.O)
        self.Y1 = self.Y - self.l1*np.sin(self.O)
        self.Y2 = self.Y + self.l2*np.sin(self.O)
        
        (self.X1,self.Y1) = self.periodic(self.X1,self.Y1)
        (self.X2,self.Y2) = self.periodic(self.X2,self.Y2)
    
    def noise_evolve(self):             # random part of s dynamics
        xiX = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau**2),self.N_ptcl) 
        xiY = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau**2),self.N_ptcl)        
        xiO = np.random.normal(0,np.sqrt(2*self.Dr*self.dt/self.tau**2),self.N_ptcl)

        self.etaX = (1-self.dt/self.tau)*self.etaX+xiX
        self.etaY = (1-self.dt/self.tau)*self.etaY+xiY
        self.etaO = (1-self.dt/self.tau)*self.etaO+xiO
#         self.etaO = xiO
        
    def force(self,X,Y,O,l,r1,r2,k1,k2,x,y):    # force and torque by x,y to X,Y with axis at angle O with length l, with force r, k
        relXx = (X.reshape(-1,1)-x.reshape(1,-1))
        relYy = (Y.reshape(-1,1)-y.reshape(1,-1))
        (relXx,relYy) = self.periodic(relXx,relYy)
        length = np.sqrt(relXx**2+relYy**2)
        
        interact1 = (length<r1)
        interact2 = (length<r2)
        
        fx     = np.sum((k1*(r1-length)*interact1 + k2*(r2-length)*interact2)*np.divide(relXx,length,out=np.zeros_like(relXx),where=length!=0), axis=1)
        
        fy     = np.sum((k1*(r1-length)*interact1 + k2*(r2-length)*interact2)*np.divide(relYy,length,out=np.zeros_like(relYy),where=length!=0), axis=1)
        
        torque = -fx*l*np.sin(O) + fy*l*np.cos(O)      # force acted on the given particle, angle 0 increase in fx=0, fy=1
        return(fx,fy,torque)

    
    def time_evolve(self):
        
        # compute force & torque
        FX = np.zeros(self.N_ptcl)
        FY = np.zeros(self.N_ptcl)
        Torque = np.zeros(self.N_ptcl)
        
        # force 1->1
        (fx,fy,torque) = self.force(self.X1,self.Y1,self.O,-self.l1,self.r1,self.r1,self.k1,self.k1,self.X1,self.Y1)
        FX     += fx
        FY     += fy
        Torque += torque
        
        # force 1->2
        (fx,fy,torque) = self.force(self.X2,self.Y2,self.O,self.l2,self.r1,self.r2,self.k1,self.k2,self.X1,self.Y1)
        FX     += fx
        FY     += fy
        Torque += torque
        
        # force 2->1
        (fx,fy,torque) = self.force(self.X1,self.Y1,self.O,-self.l1,self.r2,self.r1,self.k2,self.k1,self.X2,self.Y2)
        FX     += fx
        FY     += fy
        Torque += torque
        
        # force 2->2
        (fx,fy,torque) = self.force(self.X2,self.Y2,self.O,self.l2,self.r2,self.r2,self.k2,self.k2,self.X2,self.Y2)
        FX     += fx
        FY     += fy
        Torque += torque

        
        
        # compute noise
        self.noise_evolve()        
        
        
        # update configuration
#         forward = np.divide(FX*np.cos(self.O)+FY*np.sin(self.O),np.sqrt(FX**2+FY**2),out=np.zeros_like(FX*np.cos(self.O)+FY*np.sin(self.O)),where=(FX**2+FY**2)!=0)

        # memory in force (momentum)
        self.etaX +=FX
        self.etaY +=FY
        self.X += self.mu*(self.etaX+self.p*np.cos(self.O))*self.dt
        self.Y += self.mu*(self.etaY+self.p*np.sin(self.O))*self.dt

#         self.X += self.mu*(FX+self.etaX+self.p*np.cos(self.O))*self.dt
#         self.Y += self.mu*(FY+self.etaY+self.p*np.sin(self.O))*self.dt
        self.O += self.mur*(Torque+self.etaO)*self.dt
        
        (self.X,self.Y) = self.periodic(self.X,self.Y)
        self.set_structure()
        
    def animate(self,N_iter,directory):
        
        axrange = [-self.L/2, self.L/2, -self.L/2, self.L/2]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        os.makedirs('record/'+str(directory),exist_ok=True)


        
        
        for nn in trange(N_iter):
            ax1.clear()
            
            ax1.scatter(self.X1,self.Y1,s=self.r1*1500000/self.L**2,color='blue')
            ax1.scatter(self.X2,self.Y2,s=self.r2*1500000/self.L**2,color='red')
            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')
            fig1.canvas.draw()
            if self.record:
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
    
    
       

