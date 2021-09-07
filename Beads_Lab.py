import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys

import imageio
from PIL import Image

class Beads:     # OOP
    """basic model to simulate 2D passive objects under active noise"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,L, N_ptcl,N_active,Fs):
        
        
        # set up coefficients
        self.set_coeff(L,N_ptcl,N_active,Fs) 
      
        # initializing configuration of state
        self.set_zero()
        
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_ptcl,N_active,Fs):
        
        # system coefficients
        self.L=L
        self.N_ptcl = N_ptcl
        self.N_active = N_active
        self.Fs=Fs
        self.dt = 1/Fs
        
        self.N_skip = 50
        
        # noise coefficients
        self.D = 20
        
        
        # dynamics
        self.p = 0.1    # propulsion
        self.mu = 0.1
        self.mur = 0.01
        
        # inner structure coefficients
        self.k = 2         # epsilon of WCA potential
        self.l = 1.8    # length between fixed beads
        self.r_cut = [1,1.05,1.1]  # radius of beads
        
        
        
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        
        self.X = np.linspace(0,self.L,self.N_ptcl+1)[:self.N_ptcl]
        self.O = np.ones(self.N_active)*np.pi/2
        
        self.set_structure()
        
    def set_structure(self):
        self.Xs = self.X[1:self.N_active+1]+self.l*np.cos(self.O)   # 1~N_active
        self.Ys = self.l*np.sin(self.O)
        
        self.Xs = self.periodic(self.Xs)
    
#     def WCAx(self,rx,ry,r_cut): # return the gradient of WCA potential -> odd
#         r = np.sqrt(rx**2 + ry**2)
#         force = 4*self.k*(-12*r**(-13)/self.r_0**(-12)+6*r**(-7)/self.r_0**(-6))*(np.abs(r)<self.r_cut)
# #         return force*np.divide(rx,r,out=np.zeros_like(rx),where=r!=0)
#         return force*rx/r
    
    def WCA(self,rx,ry,r_cut):
        r_0 = r_cut*2**(-1/6)
        r = np.sqrt(rx**2 + ry**2)
        force = 4*self.k*(-12*r**(-13)/r_0**(-12)+6*r**(-7)/r_0**(-6))*(np.abs(r)<r_cut)
#         return force*np.divide(ry,r,out=np.zeros_like(ry),where=r!=0)
        return force*(rx/r,ry/r)
        
    
    def force(self):    # force from WCA potential (truncated LJ potential -> hard wall repulsion)
        f1x = np.zeros(self.N_ptcl)
        f2x = np.zeros(self.N_active)
#         f1y = np.zeros(self.N_active)
        f2y = np.zeros(self.N_active)
        
        # particle 1 constrained to wall, particle 2 rotating
        
        # relative position from -> to : r(from)-r(to)
        
        # 1->1
        relx_right = self.periodic(np.roll(self.X,-1)-self.X)
        relx_left = self.periodic(np.roll(self.X,1)-self.X)
        
        (fx,fy)=self.WCA(relx_right,0,self.r_cut[0])
        f1x += fx # right particle
        (fx,fy)=self.WCA(relx_left,0,self.r_cut[0])
        f1x += fx  # left particle
        
        # 1->2
        relx_right = self.periodic(self.X[2:self.N_active+2]-self.Xs)
        relx_left = self.periodic(self.X[:self.N_active]-self.Xs)
        rely_right = -self.Ys
        rely_left = -self.Ys
        
        (fx,fy)=self.WCA(relx_right,rely_right,self.r_cut[1])
        f2x+=fx
        f2y+=fy
        (fx,fy)=self.WCA(relx_left,rely_left,self.r_cut[1])
        f2x+=fx
        f2y+=fy
        
        # 2->1
        relx_right = self.periodic(self.Xs-self.X[:self.N_active])
        relx_left = self.periodic(self.Xs-self.X[2:self.N_active+2])
        rely_right = self.Ys
        rely_left = self.Ys
        
        (fx,fy)=self.WCA(relx_right,rely_right,self.r_cut[1])
        f1x[:self.N_active]+=fx
        (fx,fy)=self.WCA(relx_left,rely_left,self.r_cut[1])
        f1x[2:self.N_active+2]+=fx
        
        
        # 2->2
        relx_right = self.periodic(self.Xs[1:]-self.Xs[:-1])
        relx_left = self.periodic(self.Xs[:-1]-self.Xs[1:])
        rely_right = self.Ys[1:]-self.Ys[:-1]
        rely_left = self.Ys[:-1]-self.Ys[1:]
        
        (fx,fy)=self.WCA(relx_right,rely_right,self.r_cut[2])
        f2x[:-1]+=fx
        f2y[:-1]+=fy
        (fx,fy)=self.WCA(relx_left,rely_left,self.r_cut[2])
        f2x[1:]+=fx
        f2y[1:]+=fy
        
        
        # noise
        f1x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_ptcl)
        f2x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_active)
        f2y+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_active)
        
        
        # normal force
#         fN = self.p*np.sin(self.O)-f1y
        
        return(f1x,f2x,f2y)
        

    
    def time_evolve(self):
        
        # compute force & torque
        (f1x,f2x,f2y) = self.force()
        Fx = f1x
        Fx[1:self.N_active+1]+=f2x-self.p*np.cos(self.O)
        Torque = self.l/2*(f2y*np.cos(self.O)+(f1x[1:self.N_active+1]-f2x-self.p*np.cos(self.O))*np.sin(self.O))
#         print(Torque[0])
        # update configuration
        self.X+=self.mu*Fx*self.dt
        self.O+=self.mur*Torque*self.dt

        self.X = self.periodic(self.X)
        self.O = np.amax(np.vstack([self.O,np.zeros(self.N_active)]),axis=0)
        self.O = np.amin(np.vstack([self.O,(np.ones(self.N_active)*np.pi)]),axis=0)
        self.set_structure()
        
    
        
    def animate(self,N_iter,directory):
        self.time_evolve()
        
        axrange = [-self.L/2, self.L/2, -self.L/100, self.L/10]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(self.X,np.zeros(self.N_ptcl),color='red')
        ax1.scatter(self.Xs,self.Ys,color='blue')
        ax1.axis(axrange)
        ax1.set_aspect('equal','box')
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        
        os.makedirs('record/'+str(directory),exist_ok=True)
        self.time_evolve()
        
        
        for nn in trange(N_iter):
            
            ax1.clear()
            ax1.scatter(self.X,np.zeros(self.N_ptcl),s=self.r_cut[0]**2*100000/self.L**2,color='red')
            ax1.scatter(self.Xs,self.Ys,s=self.r_cut[1]**2*100000/self.L**2,color='blue')
            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')
            fig1.canvas.draw()
            if True:
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
                


        