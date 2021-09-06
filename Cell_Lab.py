d# asymmetric interacting particles under active noise
# 2D periodic boundary condition
# Yunsik Choe, Seoul national University

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys

import imageio
from PIL import Image

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
        self.p = 0
        
        # inner structure coefficients
        self.n = 3
        self.l = [-1,1,3]
        self.r = [3,2,1]
        self.k = [10,5,5]
        self.mu = 1
        self.mur = 0.2
        self.poten_order = 3
        
        self.initialize = False
        self.grid = 'ordered'
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
        
        if self.grid=='ordered':
            self.X = np.zeros(self.N_ptcl)
            self.Y = np.zeros(self.N_ptcl)
            grid = np.linspace(0,self.L,int(np.ceil(np.sqrt(self.N_ptcl)))+1)
            xgrid,ygrid = np.meshgrid(grid[:-1],grid[:-1])
            xgrid = xgrid.reshape(-1)[:self.N_ptcl]
            ygrid = ygrid.reshape(-1)[:self.N_ptcl]

            self.X = xgrid
            self.Y = ygrid
         
            
        elif self.grid =='fixed':
            lattice = self.lattice
            self.X = np.zeros(self.N_ptcl)
            self.Y = np.zeros(self.N_ptcl)
            grid = np.linspace(0,self.L,lattice+1)
            xgrid,ygrid = np.meshgrid(grid[:-1],grid[:-1])
            xgrid = xgrid.reshape(-1)
            ygrid = ygrid.reshape(-1)
            self.X[:lattice**2] = xgrid
            self.Y[:lattice**2] = ygrid
            
            grid2 = np.linspace(0,self.L/15,int(np.ceil(np.sqrt(self.N_ptcl-lattice**2)))+1)
            xgrid2,ygrid2 = np.meshgrid(grid2[:-1],grid2[:-1])
            xgrid2 = xgrid2.reshape(-1)[:self.N_ptcl-lattice**2]
            ygrid2 = ygrid2.reshape(-1)[:self.N_ptcl-lattice**2]


            self.X[lattice**2:] = xgrid2
            self.Y[lattice**2:] = ygrid2
            
            
            
            
#             grid = np.linspace(0,self.L,int(np.ceil(np.sqrt(self.N_ptcl-1)))+1)
#             xgrid,ygrid = np.meshgrid(grid[:-1],grid[:-1])
#             xgrid = xgrid.reshape(-1)[:self.N_ptcl-1]
#             ygrid = ygrid.reshape(-1)[:self.N_ptcl-1]

#             self.X[1:] = xgrid
#             self.Y[1:] = ygrid
            
#             self.X[0] = (self.X[1]+self.X[2])/2
#             self.Y[0] = self.Y[1]
        
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
#         xiO = np.random.normal(0,np.sqrt(2*self.Dr*self.dt/self.tau_noise**2),self.N_ptcl)
        xiO=0

        self.etaX = (1-self.dt/self.tau_noise)*self.etaX+xiX
        self.etaY = (1-self.dt/self.tau_noise)*self.etaY+xiY
#         self.etaO = (1-self.dt/self.tau_noise)*self.etaO+xiO
        self.etaO = xiO
        
    def force(self,i,j):    # force and torque by x,y to X,Y with axis at angle O with length l, with force r, k
        relXx = (self.Xs[i].reshape(-1,1)-self.Xs[j].reshape(1,-1))
        relYy = (self.Ys[i].reshape(-1,1)-self.Ys[j].reshape(1,-1))
        
        
        (relXx,relYy) = self.periodic(relXx,relYy)
        length = np.sqrt(relXx**2+relYy**2)
        
        interact1 = (length<self.r[i])
        interact2 = (length<self.r[j])
        
        fx     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relXx,length,out=np.zeros_like(relXx),where=length!=0), axis=1)
        
        fy     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relYy,length,out=np.zeros_like(relYy),where=length!=0), axis=1)
        
        torque = -fx*self.l[i]*np.sin(self.O) + fy*self.l[i]*np.cos(self.O)      # force acted on the given particle, angle 0 increase in fx=0, fy=1
        return(fx,fy,torque)

    
    def time_evolve(self):
        
        # compute force & torque
        FX = np.zeros(self.N_ptcl)
        FY = np.zeros(self.N_ptcl)
        Torque = np.zeros(self.N_ptcl)
        
#         if self.grid =='fixed':
#             FX = np.zeros(self.N_ptcl)
#             FY = np.zeros(self.N_ptcl)
#             Torque = np.zeros(self.N_ptcl)
            
#             for i in range(self.n):
#                 for j in range(self.n):
#                     relXx = self.Xs[i,0]-self.Xs[j]
#                     relYy = self.Ys[i,0]-self.Ys[j]
        
        
#                     (relXx,relYy) = self.periodic(relXx,relYy)
#                     length = np.sqrt(relXx**2+relYy**2)

#                     interact1 = (length<self.r[i])
#                     interact2 = (length<self.r[j])

#                     fx     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relXx,length,out=np.zeros_like(relXx),where=length!=0))

#                     fy     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relYy,length,out=np.zeros_like(relYy),where=length!=0))

#                     torque = -fx*self.l[i]*np.sin(self.O[0]) + fy*self.l[i]*np.cos(self.O[0])      # force acted on the given particle, angle 0 increase in fx=0, fy=1
#                     FX[0]+=fx
#                     FY[0]+=fy
#                     Torque[0]+=torque
            
#         else:
        for i in range(self.n):
            for j in range(self.n):
                (fx,fy,torque) = self.force(i,j)
                FX     += fx
                FY     += fy
                Torque += torque


        # compute noise
        self.noise_evolve()        
        
        


        # memory in force (momentum)
#         self.FX     = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  FX
#         self.FY     = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  FY
#         self.Torque = (1-self.dt/self.tau_momentum)*self.FX*self.memory  +  Torque
        
        
        
        # update configuration


        
        
        self.X += self.mu*(FX+self.etaX)*self.dt
        self.Y += self.mu*(FY+self.etaY)*self.dt

#         self.X += self.mu*(FX+self.etaX+self.p*np.cos(self.O))*self.dt
#         self.Y += self.mu*(FY+self.etaY+self.p*np.sin(self.O))*self.dt
        
    
        self.O += self.mur*(Torque+self.etaO)*self.dt
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
    
    
def simulate(order, name):

    lattice = 16
    C1 = Cell_Lab(L=24,N_ptcl=lattice**2+6**2,Fs=500)

    C1.lattice = lattice
    C1.D = 50
    C1.Dr = 0
    C1.tau_noise =0.05

    if order ==3:
        mul = 1
    elif order ==2:
        mul = 0.125

    C1.n = 5
    C1.l = [ -1.3,-1,  -0.4,   0.2,0.6 ]
    C1.r = [0.45,    0.45,     0.45,0.45,0.45]
    C1.k = [1500*mul,    1300*mul,  1000*mul,1500*mul,1000*mul]




    C1.poten_order = int(order)
    jump = 6

    C1.mu = 0.00004*np.ones(C1.N_ptcl)
    C1.mur = 0.00002*np.ones(C1.N_ptcl)


    C1.mu[lattice**2:] = 0.1
    C1.mur[lattice**2:] = 0.05



    C1.N_skip = 100
    C1.p = 0

#     name = '_test3'
    N_simul = 3000

    C1.record = True           # True, False
    C1.initialize = False      # True, False
    C1.grid = 'fixed'#'fixed'     # 'ordered','fixed',False
    C1.ang_ordered = 'parallel'     # 'parallel', 'anti-parallel', False


    C1.set_zero()
    # C1.O[::jump]+=np.pi
    C1.O[lattice**2:]+=np.pi


    C1.animate(N_simul,name)
    path_dir = os.getcwd()+'/record/'+name+'/'
    t_list = np.arange(N_simul)
    # t_list = np.arange(500)
    path=[path_dir+f"{t}.png" for t in t_list]
    paths=[Image.open(i) for i in path]
    imageio.mimsave(path_dir+'anim'+name+'.gif',paths,fps=30)

    
    
class Cell_Lattice_Lab: 
    """current evaluation in non-interacting ptcls in lattice potential"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,L, N_mobile, N_view,Fs):
        
        
        # set up coefficients
        self.set_coeff(L,N_mobile, N_view,Fs) 
      
        # initializing configuration of state
        self.set_zero()
        
        self.record = False
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_mobile, N_view,Fs):
        
        # system coefficients
        self.L=L
        self.N_view = N_view
        self.N_mobile = N_mobile
        self.N_lattice = 2
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
        self.p = 0
        
        # inner structure coefficients
        self.n = 3
        self.l = [-1,1,3]
        self.r = [3,2,1]
        self.k = [10,5,5]
        self.mu = 1
        self.mur = 0.2
        self.poten_order = 3
        
        self.initialize = False
        self.grid = 'ordered'
        self.ang_ordered = False
        

                  
        
        
    # boundary condition
    
    def periodic(self,x,y,multiplicity):             # add periodic boundary condition using modulus function
        mod_x = -self.L*multiplicity/2   +    (x+self.L*multiplicity/2)%(self.L*multiplicity)               # returns -L/2 ~ L/2
        mod_y = -self.L*multiplicity/2   +    (y+self.L*multiplicity/2)%(self.L*multiplicity)               # returns -L/2 ~ L/2

        return (mod_x,mod_y)
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.etaX = np.random.normal(0,np.sqrt(self.D/self.tau_noise),self.N_mobile) 
        self.etaY = np.random.normal(0,np.sqrt(self.D/self.tau_noise),self.N_mobile) 
        self.etaO = np.random.normal(0,np.sqrt(self.Dr/self.tau_noise),self.N_mobile)
        
        # lattice particles
        self.X_lattice = np.zeros(self.N_lattice**2)
        self.Y_lattice = np.zeros(self.N_lattice**2)
        self.O_lattice = np.ones(self.N_lattice**2)*np.pi/6
        grid = np.linspace(0,self.L,self.N_lattice+1)
        xgrid,ygrid = np.meshgrid(grid[:-1],grid[:-1])
        xgrid = xgrid.reshape(-1)
        ygrid = ygrid.reshape(-1)

        self.X_lattice = xgrid
        self.Y_lattice = ygrid 
        
        
        self.Xs_lattice = self.X_lattice.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.cos(self.O_lattice)
        self.Ys_lattice = self.Y_lattice.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.sin(self.O_lattice)
        (self.Xs_lattice,self.Ys_lattice) = self.periodic(self.Xs_lattice,self.Ys_lattice,1)

        # view setting
        X_lattice_view = np.zeros((self.N_lattice*self.N_view)**2)
        Y_lattice_view = np.zeros((self.N_lattice*self.N_view)**2)
        self.O_lattice_view = np.ones((self.N_lattice*self.N_view)**2)*np.pi/6
        grid_view = np.linspace(0,self.L*self.N_view,self.N_lattice*self.N_view+1)
        xgrid_view,ygrid_view = np.meshgrid(grid_view[:-1],grid_view[:-1])
        xgrid_view = xgrid_view.reshape(-1)
        ygrid_view = ygrid_view.reshape(-1)

        X_lattice_view = xgrid_view
        Y_lattice_view = ygrid_view
        
        
        self.Xs_lattice_view = X_lattice_view.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.cos(self.O_lattice_view)
        self.Ys_lattice_view = Y_lattice_view.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.sin(self.O_lattice_view)
        (self.Xs_lattice_view,self.Ys_lattice_view) = self.periodic(self.Xs_lattice_view,self.Ys_lattice_view,self.N_view)
        
        # mobile particles
        self.X_mobile = np.zeros(self.N_mobile)
        self.Y_mobile = np.zeros(self.N_mobile)
        self.O_mobile = np.ones(self.N_mobile)*np.pi*7/6
        
        self.set_structure()
        
    def set_structure(self):
        self.Xs_mobile = self.X_mobile.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.cos(self.O_mobile)
        self.Ys_mobile = self.Y_mobile.reshape(1,-1) + np.array(self.l).reshape(-1,1)*np.sin(self.O_mobile)
        
#         (self.Xs_mobile,self.Ys_mobile) = self.periodic(self.Xs_mobile,self.Ys_mobile)
    
    def noise_evolve(self):             # random part of s dynamics
        xiX = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau_noise**2),self.N_mobile) 
        xiY = np.random.normal(0,np.sqrt(2*self.D*self.dt/self.tau_noise**2),self.N_mobile)        
#         xiO = np.random.normal(0,np.sqrt(2*self.Dr*self.dt/self.tau_noise**2),self.N_ptcl)
        xiO=0

        self.etaX = (1-self.dt/self.tau_noise)*self.etaX+xiX
        self.etaY = (1-self.dt/self.tau_noise)*self.etaY+xiY
#         self.etaO = (1-self.dt/self.tau_noise)*self.etaO+xiO
        self.etaO = xiO
        
    def force(self,i,j):    # force and torque by x,y to X,Y with axis at angle O with length l, with force r, k
        relXx = (self.Xs_mobile[i].reshape(-1,1)-self.Xs_lattice[j].reshape(1,-1))
        relYy = (self.Ys_mobile[i].reshape(-1,1)-self.Ys_lattice[j].reshape(1,-1))
        
        
        (relXx,relYy) = self.periodic(relXx,relYy,1)
        length = np.sqrt(relXx**2+relYy**2)
        
        interact1 = (length<self.r[i])
        interact2 = (length<self.r[j])
        
        fx     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relXx,length,out=np.zeros_like(relXx),where=length!=0), axis=1)
        
        fy     = np.sum((self.k[i]*interact1*(self.r[i]-length)**(self.poten_order-1) + self.k[j]*interact2*(self.r[j]-length)**(self.poten_order-1))*np.divide(relYy,length,out=np.zeros_like(relYy),where=length!=0), axis=1)
        
        torque = -fx*self.l[i]*np.sin(self.O_mobile) + fy*self.l[i]*np.cos(self.O_mobile)      # force acted on the given particle, angle 0 increase in fx=0, fy=1
        return(fx,fy,torque)

    
    def time_evolve(self):
        
        # compute force & torque
        FX = np.zeros(self.N_mobile)
        FY = np.zeros(self.N_mobile)
        Torque = np.zeros(self.N_mobile)

        for i in range(self.n):
            for j in range(self.n):
                (fx,fy,torque) = self.force(i,j)
                FX     += fx
                FY     += fy
                Torque += torque


        # compute noise
        self.noise_evolve()        
        
        
        # update configuration

        self.X_mobile += self.mu*(FX+self.etaX)*self.dt
        self.Y_mobile += self.mu*(FY+self.etaY)*self.dt
        self.O_mobile += self.mur*(Torque+self.etaO)*self.dt
        
#         (self.X_mobile,self.Y_mobile) = self.periodic(self.X_mobile,self.Y_mobile)
        self.set_structure()
    
    def measure(self,N_iter):
        
    
    
    def animate(self,N_iter,directory):
        axrange = [-self.L*self.N_view/2, self.L*self.N_view/2, -self.L*self.N_view/2, self.L*self.N_view/2]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        for i in range(self.n):
            ax2.scatter(0,self.l[i],s=self.r[i]*500000/(self.L*self.N_view)**2)
        ax2.quiver(0,self.l[0],0,(self.l[-1]-self.l[0]),scale =self.L*self.N_view)
        ax2.axis(axrange)
        ax2.set_aspect('equal','box')
        ax2.grid()
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        
        os.makedirs('record/'+str(directory),exist_ok=True)


        self.time_evolve()


        
        for nn in trange(N_iter):
            
            if self.record:
                ax1.clear()
                ax1.quiver(self.Xs_lattice_view[0],self.Ys_lattice_view[0],(self.l[-1]-self.l[0])*np.cos(self.O_lattice_view),(self.l[-1]-self.l[0])*np.sin(self.O_lattice_view),scale = self.L*self.N_view,color='red')
                (self.Xs_mobile_view,self.Ys_mobile_view) = self.periodic(self.Xs_mobile,self.Ys_mobile,self.N_view)
                ax1.quiver(self.Xs_mobile_view[0],self.Ys_mobile_view[0],(self.l[-1]-self.l[0])*np.cos(self.O_mobile),(self.l[-1]-self.l[0])*np.sin(self.O_mobile),scale = self.L*self.N_view,color='blue')
                ax1.axis(axrange)
                ax1.set_aspect('equal', 'box')
                ax1.grid()
                fig1.canvas.draw()
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
    