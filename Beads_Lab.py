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
    
    def __init__(self,L, N_ptcl,N_active,N_ensemble,Fs,g):
        self.boundary='periodic'


        
        # set up coefficients
        self.set_coeff(L,N_ptcl,N_active,N_ensemble,Fs,g) 
      
        # initializing configuration of state
        self.X = np.ones(self.N_ensemble).reshape(-1,1)*np.linspace(0,self.L,self.N_ptcl+1)[:self.N_ptcl].reshape(1,-1)
        self.O = np.ones((self.N_ensemble,self.N_active))*np.pi/2
        self.set_zero((np.ones(N_ensemble)==np.ones(N_ensemble)))
        
        
        
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_ptcl,N_active,N_ensemble,Fs,g):
        
        # system coefficients
        self.L=L
        self.N_ptcl = N_ptcl
        self.N_active = N_active
        self.N_ensemble = N_ensemble
        self.Fs=Fs
        self.dt = 1/Fs
        
        self.N_skip = 50
        
        # noise coefficients
#         self.D = 5000
        
        
        # dynamics
        self.p = 2000    # propulsion
        self.mu = 0.002    #1/gamma
#         self.mur = 0.002
        self.kT = 1
        
        # inner structure coefficients
        self.k1 = 1         # epsilon of WCA potential
        self.k2 = 0.1
        self.l = 1    # length between fixed beads
        self.r_cut = [1.5,1.65,1.8]  # radius of beads [r1+r1,r1+r2,r2+r2]
        self.g = g
        
        
        
        self.Omin = np.arcsin((self.r_cut[2]-self.r_cut[1])/self.l)
        
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self,flag):              # initializing simulation configurations
        
        if self.boundary=='periodic':
            l0 = self.r_cut[0]*(2.5)
            l1 = 2*self.r_cut[0]*(self.N_ptcl-self.N_active)
            l2 = 2*self.r_cut[2]*(2.5)/2*(self.N_active)
            seg1 = np.ones(np.sum(flag)).reshape(-1,1)*np.linspace(self.L*(l0/(l0+l1+l2)),self.L*(l0+l2)/(l0+l1+l2),self.N_active+1).reshape(1,-1)
            seg2 = np.ones(np.sum(flag)).reshape(-1,1)*np.linspace(self.L*(l0+l2)/(l0+l1+l2),self.L,self.N_ptcl-self.N_active+1).reshape(1,-1)
            self.X[flag,0] = (1/2)*(seg2[:,-1]+seg2[:,-2])
            self.X[flag,1:self.N_active+1] = (1/2)*(seg1[:,1:]+seg1[:,:-1])# active ones
            self.X[flag,self.N_active+1:] = (1/2)*(seg2[:,1:-1]+seg2[:,:-2])     # passive ones
        else:
            self.X[flag,1:self.N_active+1] =np.ones(np.sum(flag)).reshape(-1,1)*np.arange(self.N_active).reshape(1,-1)*2.5*self.r_cut[2] # active ones
            self.X[flag,self.N_active+1:] =  np.ones(np.sum(flag)).reshape(-1,1)*np.arange(self.N_ptcl-self.N_active-1).reshape(1,-1)*2.5*self.r_cut[0] + (self.N_active)*2.5*self.r_cut[2] # passive ones
            self.X[flag,0] = (self.N_active)*2.5*self.r_cut[2]+(self.N_ptcl-self.N_active-1)*2.5*self.r_cut[0]


        
        self.O[flag] = np.ones((np.sum(flag),self.N_active))*np.pi*(1/2+1/10)
        self.O[flag,0] = np.pi
        self.O[flag,-1] = np.pi/2
        
        self.set_structure()
        
        
    def set_structure(self):
        self.Xs = self.X[:,1:self.N_active+1]+self.l*np.cos(self.O)   # 1~N_active
        self.Ys = self.l*np.sin(self.O)
        
        self.Xs = self.periodic(self.Xs)
    
#     def WCAx(self,rx,ry,r_cut): # return the gradient of WCA potential -> odd
#         r = np.sqrt(rx**2 + ry**2)
#         force = 4*self.k*(-12*r**(-13)/self.r_0**(-12)+6*r**(-7)/self.r_0**(-6))*(np.abs(r)<self.r_cut)
# #         return force*np.divide(rx,r,out=np.zeros_like(rx),where=r!=0)
#         return force*rx/r
    
    def WCA(self,rx,ry,r_cut,k):
        r_0 = r_cut*2**(-1/6)
        r = np.sqrt(rx**2 + ry**2)
        force = 4*k*(-12*r**(-13)/r_0**(-12)+6*r**(-7)/r_0**(-6))*(np.abs(r)<r_cut)
#         return force*np.divide(ry,r,out=np.zeros_like(ry),where=r!=0)
#         return force*(np.divide(rx,r,out=np.zeros_like(rx),where=r!=0),np.divide(ry,r,out=np.zeros_like(ry),where=r!=0))
        return force*(rx/r,ry/r)
    
    def force(self):    # force from WCA potential (truncated LJ potential -> hard wall repulsion)
        f1x = np.zeros((self.N_ensemble,self.N_ptcl))
        f2x = np.zeros((self.N_ensemble,self.N_active))
#         f1y = np.zeros(self.N_active)
        f2y = np.zeros((self.N_ensemble,self.N_active))
        
        # particle 1 constrained to wall, particle 2 rotating
        
        # relative position from -> to : r(from)-r(to)
        
        # 1->1
        relx_right = np.roll(self.X,-1,axis=1)-self.X
        if self.boundary=='periodic':
            relx_right=self.periodic(relx_right)
        relx_left = np.roll(self.X,1,axis=1)-self.X
        if self.boundary=='periodic':
            relx_left=self.periodic(relx_left)
        rely_right = np.zeros((self.N_ensemble,self.N_ptcl))
        rely_left = np.zeros((self.N_ensemble,self.N_ptcl))
        rely_right[:,0] = -self.r_cut[0]+self.r_cut[1]
        rely_left[:,self.N_active+2] = self.r_cut[0]-self.r_cut[1]
        
        lengthR = np.concatenate(([self.r_cut[0]+self.r_cut[1]],(self.r_cut[1]*2)*np.ones(self.N_active-1),[self.r_cut[0]+self.r_cut[1]],(self.r_cut[0]*2)*np.ones(self.N_ptcl-self.N_active-1)))
        lengthL = np.concatenate(([self.r_cut[0]*2],[self.r_cut[0]+self.r_cut[1]],(self.r_cut[1]*2)*np.ones(self.N_active-1),[self.r_cut[0]+self.r_cut[1]],(self.r_cut[0]*2)*np.ones(self.N_ptcl-self.N_active-2)))
        
        (fx,fy)=self.WCA(relx_right,rely_right,lengthR,self.k1)
        f1x += fx # right particle
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL,self.k1)
        f1x += fx  # left particle
        
        
        # 1->2
        relx_right = self.X[:,2:self.N_active+2]-self.Xs
        if self.boundary=='periodic':
            relx_right=self.periodic(relx_right)
        relx_left = self.X[:,:self.N_active]-self.Xs
        if self.boundary=='periodic':
            relx_left=self.periodic(relx_left)
        rely_right = -self.Ys
        rely_left = -self.Ys
        rely_right[:,-1] += self.r_cut[0]-self.r_cut[1]
        rely_left[:,0] -= self.r_cut[0]-self.r_cut[1]
        
        lengthR = np.concatenate(((self.r_cut[1]+self.r_cut[2])*np.ones(self.N_active-1),[(self.r_cut[0]+self.r_cut[2])]))
        lengthL = np.concatenate(([(self.r_cut[0]+self.r_cut[2])],(self.r_cut[1]+self.r_cut[2])*np.ones(self.N_active-1)))
        
        (fx,fy)=self.WCA(relx_right,rely_right,lengthR,self.k2)
        f2x+=fx
        f2y+=fy
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL,self.k2)
        f2x+=fx
        f2y+=fy
        
        
        # 2->1
        
        relx_right = self.Xs-self.X[:,:self.N_active]
        if self.boundary=='periodic':
            relx_right=self.periodic(relx_right)
        relx_left = self.Xs-self.X[:,2:self.N_active+2]
        if self.boundary=='periodic':
            relx_left=self.periodic(relx_left)
        rely_right = self.Ys
        rely_left = self.Ys
        rely_right[:,0] -= self.r_cut[0]-self.r_cut[1]
        rely_left[:,-1] += self.r_cut[0]-self.r_cut[1]
        
        lengthR = np.concatenate(([(self.r_cut[0]+self.r_cut[2])],(self.r_cut[1]+self.r_cut[2])*np.ones(self.N_active-1)))
        lengthL = np.concatenate(((self.r_cut[1]+self.r_cut[2])*np.ones(self.N_active-1),[(self.r_cut[0]+self.r_cut[2])]))
        
        (fx,fy)=self.WCA(relx_right,rely_right,lengthR,self.k2)
        f1x[:,:self.N_active]+=fx
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL,self.k2)
        f1x[:,2:self.N_active+2]+=fx
        
        
        # 2->2
        relx_right = self.Xs[:,1:] -self.Xs[:,:-1]
        if self.boundary=='periodic':
            relx_right=self.periodic(relx_right)
        relx_left = self.Xs[:,:-1]  -self.Xs[:,1:]
        if self.boundary=='periodic':
            relx_left=self.periodic(relx_left)
        rely_right = self.Ys[:,1:]-self.Ys[:,:-1]
        rely_left = self.Ys[:,:-1]-self.Ys[:,1:]
        
        (fx,fy)=self.WCA(relx_right,rely_right,self.r_cut[2]*2,self.k2)
        f2x[:,:-1]+=fx
        f2y[:,:-1]+=fy
        (fx,fy)=self.WCA(relx_left,rely_left,self.r_cut[2]*2,self.k2)
        f2x[:,1:]+=fx
        f2y[:,1:]+=fy
        
        
        # noise
        f1x+=np.random.normal(0,np.sqrt(2*self.kT/(self.mu*self.dt)),(self.N_ensemble,self.N_ptcl))
        f2x+=np.random.normal(0,np.sqrt(2*self.kT/(self.mu*self.dt)),(self.N_ensemble,self.N_active))
        f2y+=np.random.normal(0,np.sqrt(2*self.kT/(self.mu*self.dt)),(self.N_ensemble,self.N_active))
        
        # normal force
#         fN = self.p*np.sin(self.O)-f1y

        # gravity
        f2y-=self.g
        
        return(f1x,f2x,f2y)
        

    
    def time_evolve(self):
        if self.boundary!='periodic':
            self.X-=self.V*self.dt
        
        (f1x,f2x,f2y) = self.force()
        
        # passive particles
        self.X[:,0]+= self.mu*self.dt*f1x[:,0]
        self.X[:,self.N_active+1:] += self.mu*self.dt*f1x[:,self.N_active+1:]
        
        
        
        # active particles
        
        Fx = f1x[:,1:self.N_active+1]+f2x - self.p*np.cos(self.O)/self.mu
        
#         self.v = np.mean(Fx,axis=1)*self.mu
        self.v = np.sum(np.cos(self.O),axis=1)  #*self.mu
    
        Torque = self.l*(f2y*np.cos(self.O)-f2x*np.sin(self.O))
        
        
        # update configuration
        dx = (self.l**2*Fx/self.mu + self.l*np.sin(self.O)*Torque/self.mu)/(self.l**2*(1+np.cos(self.O)**2)/self.mu**2)
        do = (self.l*Fx*np.sin(self.O)/self.mu + 2*Torque/self.mu)/(self.l**2*(1+np.cos(self.O)**2)/self.mu**2)
        
        self.X[:,1:self.N_active+1]+=dx*self.dt
        self.O+=do*self.dt
        
        
        if self.boundary=='periodic':
            self.v = np.sum(np.cos(self.O),axis=1)  #*self.mu
            self.X = self.periodic(self.X)
        else:
            self.X[:,self.N_active] = self.XR
        self.O = np.amax(np.vstack([[self.O],[np.ones((self.N_ensemble,self.N_active))*self.Omin]]),axis=0)
        self.O = np.amin(np.vstack([[self.O],[(np.ones((self.N_ensemble,self.N_active))*(np.pi-self.Omin))]]),axis=0)  


        self.set_structure()
        
    
        
    def animate(self,N_iter,directory):
        self.set_zero((np.ones(self.N_ensemble)==np.ones(self.N_ensemble)))
        if self.boundary=='periodic':
            axrange = [-self.L/2, self.L/2, -self.L/100, self.L/10]
        else:
            axrange = [0,self.L,-self.L/100, self.L/10]
        #Setup plot for updated positions
        fig1 = plt.figure(figsize=(6,8))
        ax1 = fig1.add_subplot(311)
        ax2 = fig1.add_subplot(312)
        ax3 = fig1.add_subplot(313)


        ax1.scatter(self.X[0],np.zeros(self.N_ptcl),color='red',edgecolors = 'black')
        ax1.scatter(self.Xs[0],self.Ys[0],color='blue')
        ax1.axis(axrange)
        ax1.set_aspect('equal','box')
        
        ax2.scatter(self.X[1],np.zeros(self.N_ptcl),color='red',edgecolors = 'black')
        ax2.scatter(self.Xs[1],self.Ys[0],color='blue')
        ax2.axis(axrange)
        ax2.set_aspect('equal','box')
        
        ax3.scatter(self.X[2],np.zeros(self.N_ptcl),color='red',edgecolors = 'black')
        ax3.scatter(self.Xs[2],self.Ys[0],color='blue')
        ax3.axis(axrange)
        ax3.set_aspect('equal','box')
        
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        
        os.makedirs('record/'+str(directory),exist_ok=True)
        self.time_evolve()
        Othres = self.Omin+np.pi/50
        
        
        for nn in trange(N_iter):
            
            ax1.clear()
            ax1.scatter(self.X[0][0],self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax1.scatter(self.X[0][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax1.scatter(self.X[0][1:self.N_active+1],np.zeros(self.N_active)+self.r_cut[1],s=self.r_cut[1]**2*200000/self.L**2,color='green')

            ax1.scatter(self.Xs[0],self.Ys[0]+self.r_cut[1],s=self.r_cut[2]**2*200000/self.L**2,color='blue')
            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')
            
            
            
            ax2.clear()
            ax2.scatter(self.X[1][0],self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax2.scatter(self.X[1][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax2.scatter(self.X[1][1:self.N_active+1],np.zeros(self.N_active)+self.r_cut[1],s=self.r_cut[1]**2*200000/self.L**2,color='green')
            ax2.scatter(self.Xs[1],self.Ys[1]+self.r_cut[1],s=self.r_cut[2]**2*200000/self.L**2,color='blue')
            ax2.axis(axrange)
            ax2.set_aspect('equal', 'box')
            
            
            
            ax3.clear()
            ax3.scatter(self.X[2][0],self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax3.scatter(self.X[2][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_cut[0],s=self.r_cut[0]**2*200000/self.L**2,color='red')
            ax3.scatter(self.X[2][1:self.N_active+1],np.zeros(self.N_active)+self.r_cut[1],s=self.r_cut[1]**2*200000/self.L**2,color='green')
            ax3.scatter(self.Xs[2],self.Ys[2]+self.r_cut[1],s=self.r_cut[2]**2*200000/self.L**2,color='blue')
            ax3.axis(axrange)
            ax3.set_aspect('equal', 'box')
            
            
            
            
            fig1.canvas.draw()
            if True:
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
            stuck = (np.cos(self.O[:,0])<-np.cos(Othres))*(np.cos(self.O[:,-1])>np.cos(Othres))
            self.set_zero(stuck)
                
    def measure(self,N_iter,initialize):
        if initialize:
            self.set_zero()
        v_traj = np.zeros((self.N_ensemble,int(N_iter)))
        
        for j in trange(int(N_iter)):
            self.time_evolve()
            v_traj[:,j] = self.v
        v = np.abs(np.amax(v_traj[-int():],axis=1))
        bins = np.linspace(-2,2,100)
        count_sum = np.zeros(99)
        for i in range(self.N_ensemble):
            count,_,_=plt.hist(v_traj[i], bins)
            count_sum+=count
        plt.plot((bins[:-1]+bins[1:])/2,count_sum)
        plt.show()
        
#         return((bins[:-1]+bins[1:])/2,count_sum)
#         v = np.abs(np.average(v_traj,axis=1))
        return v_traj


    def transit(self,N_iter):
        self.set_zero((np.ones(self.N_ensemble)==np.ones(self.N_ensemble)))
        
        move_in = [np.zeros(0)]*self.N_ensemble
        move_out = [np.zeros(0)]*self.N_ensemble
        count = 0
        age = np.zeros(self.N_ensemble,dtype = np.int)

        self.time_evolve()
        
        Othres = self.Omin+np.pi/50
        
        
        # starting with right moving
        
#         prev_right = (np.cos(self.O[:,0])<-np.cos(Othres))*(~(np.cos(self.O[:,-1])>np.cos(Othres)))
#         prev_left = (np.cos(self.O[:,-1])>np.cos(Othres))*(~(np.cos(self.O[:,0])<-np.cos(Othres)))
#         prev_stuck = (np.cos(self.O[:,0])<-np.cos(Othres))*(np.cos(self.O[:,-1])>np.cos(Othres))
        time = 0
        
        for i in range(self.N_ensemble):
            count+=1
            move_in[i] = np.append(move_in[i],time)
#             if prev_right[i]:
#                 move_in[i] = np.append(move_in[i],time)
#                 count+=1
#             elif prev_left[i]:
#                 move_in[i] = np.append(move_in[i],time)
#                 count+=1

        v1_t = np.zeros(N_iter)
        v2_t = np.zeros(N_iter)
        for j in trange(N_iter):
            self.time_evolve()
            event = (np.cos(self.O[:,-1])>np.cos(Othres))
            
#             right = (np.cos(self.O[:,0])<-np.cos(Othres))*(~(np.cos(self.O[:,-1])>np.cos(Othres)))
#             left = (np.cos(self.O[:,-1])>np.cos(Othres))*(~(np.cos(self.O[:,0])<-np.cos(Othres)))
#             stuck = (np.cos(self.O[:,0])<-np.cos(Othres))*(np.cos(self.O[:,-1])>np.cos(Othres))
        
            time = j*self.dt#*N_time
            
    
    
                    
            for i in range(self.N_ensemble):
                if event[i]:
                    move_out[i] = np.append(move_out[i],time)
                    move_in[i] = np.append(move_in[i],time)
                    age[i] = 0
                    count +=1
                else:
                    v1_t[age[i]]+=np.abs(self.v[i])
                    v2_t[age[i]]+=self.v[i]**2
                    age[i] +=1
                
                
#                 if right[i]:
#                     if(not prev_right[i]):
#                         move_in[i] = np.append(move_in[i],time)
#                         age[i] = 0
#                         count+=1
#                     v1_t[age[i]]+=np.abs(self.v[i])
#                     v2_t[age[i]]+=self.v[i]**2
#                 elif left[i]:
#                     if (not prev_left[i]):
#                         move_in[i] = np.append(move_in[i],time)
#                         age[i] = 0   
#                         count+=1
#                     v1_t[age[i]]+=np.abs(self.v[i])
#                     v2_t[age[i]]+=self.v[i]**2
#                 elif stuck[i]*(prev_right[i] or prev_left[i]):
#                     move_out[i] = np.append(move_out[i],time)
            self.set_zero(event)
            age +=1
            
                    
#             prev_right = right
#             prev_left = left
#             prev_stuck = stuck
            
            
        time +=self.dt
        for i in range(self.N_ensemble):
            move_out[i] = np.append(move_out[i],time)
#             if right[i]:
#                 move_out[i] = np.append(move_out[i],time)
#             elif left[i]:
#                 move_out[i] = np.append(move_out[i],time)
        
        
        v_t_avg = v1_t/count
        v_t_var = v2_t/count-(v1_t/count)**2
        
        
#             elif stuck[i]:
#                 stuck_out[i] = np.append(stuck_out[i],time)

#         return(right_in,left_in,stuck_in, right_out, left_out,stuck_out)

        return(move_in, move_out,v_t_avg,v_t_var)


          
            
        
def time(N_ptcl, N_active,g):

    
    B1 = Beads(L=68, N_ptcl = N_ptcl,N_active = N_active,N_ensemble = 300,Fs=500,g=g)

    B1.boundary='periodic'
    B1.p = 50
#     B1.D = D  #20
    B1.mu = 0.2
#     B1.mur = 0.03
    B1.k1 = 1
    B1.k2 = 1
    B1.kT = 1
    B1.r_cut = [1.3,1.55,1.8]
    # B1.r_cut = [1.3,0.8,0.9]
    B1.l = 1.3
    B1.Omin = 0


    B1.L = ((B1.N_ptcl-B1.N_active)*B1.r_cut[0]+(B1.N_active)*B1.r_cut[2]+2*B1.r_cut[2])*1.5

    direc = '211229_v_t/N_ptcl='+str(B1.N_ptcl)+',g='+str(B1.g)
    os.makedirs(direc,exist_ok=True)



#     (right_in,left_in,stuck_in, right_out, left_out,stuck_out) = B1.transit(200000)
    N_simul = 1000000
    (move_in,move_out,v_t_avg,v_t_var) = B1.transit(N_simul)


#     right_in =np.array(right_in)
#     left_in = np.array(left_in)
#     stuck_in = np.array(stuck_in)
#     right_out =np.array(right_out)
#     left_out = np.array(left_out)
#     stuck_out = np.array(stuck_out)
    move_in = np.array(move_in,dtype=object)
    move_out =np.array(move_out,dtype=object)

    save_dict={}
#     save_dict['right_in'] = right_in
#     save_dict['left_in'] = left_in
#     save_dict['stuck_in'] = stuck_in
#     save_dict['right_out'] = right_out
#     save_dict['left_out'] = left_out
#     save_dict['stuck_out'] = stuck_out
    save_dict['move_in'] = move_in
    save_dict['move_out'] = move_out 
    
    save_dict['N_ens'] = B1.N_ensemble
    save_dict['N_total'] = B1.N_ptcl
    save_dict['N_active'] = B1.N_active
    save_dict['dt'] = B1.dt
    save_dict['N_simul'] = N_simul
    save_dict['v_t_avg'] = v_t_avg
    save_dict['v_t_var'] = v_t_var
    
    np.savez(direc+'/N'+str(B1.N_active)+'.npz', **save_dict)


def v_time(N_passive, N_active,g,D,V_init,V_fin,N_V):

    B1 = Beads(L=68, N_ptcl = N_active+N_passive,N_active =N_active,N_ensemble = 100,Fs=2000,g=g)
    B1.boundary='non-periodic'
    # B1.boundary='periodic'


    B1.p =20
    B1.D = D
    B1.mu = 0.5
    B1.mur =0.05
    B1.Omin = 0
    B1.k1 = 1
    B1.k2 = 1
    B1.r_cut = [1.5,1.65,1.8]
    B1.cr = 1


    B1.L = ((B1.N_ptcl-B1.N_active)*B1.r_cut[0]+(B1.N_active)*B1.r_cut[2]+2*B1.r_cut[2])*1.2
    l0 = B1.r_cut[2]*2
    l1 = B1.r_cut[0]*(B1.N_ptcl-B1.N_active)
    l2 = B1.r_cut[2]*2.5/2*(B1.N_active)
    seg1 = np.linspace(B1.L*(l0/(l0+l1+l2)),B1.L*(l0+l2)/(l0+l1+l2),B1.N_active+1)
    B1.XR = (1/2)*(seg1[-1]+seg1[-2])
    
    direc = '211126_v_time/N_passive='+str(N_passive)+',g='+str(B1.g)+',D='+str(B1.D)+'.N='+str(B1.N_active)
    os.makedirs(direc,exist_ok=True)



    V_axis = np.linspace(V_init,V_fin,N_V)
    for V in V_axis:
        B1.V = V
        B1.set_zero((np.ones(B1.N_ensemble)==np.ones(B1.N_ensemble)))



        #     (right_in,left_in,stuck_in, right_out, left_out,stuck_out) = B1.transit(200000)
        N_simul = 3000000
        (move_in,move_out,v_t_avg,v_t_var) = B1.transit(N_simul)


        move_in = np.array(move_in,dtype=object)
        move_out =np.array(move_out,dtype=object)

        save_dict={}
        save_dict['move_in'] = move_in
        save_dict['move_out'] = move_out 

        save_dict['N_ens'] = B1.N_ensemble
        save_dict['N_total'] = B1.N_ptcl
        save_dict['N_active'] = B1.N_active
        save_dict['dt'] = B1.dt
        save_dict['N_simul'] = N_simul
        save_dict['v_t_avg'] = v_t_avg
        save_dict['v_t_var'] = v_t_var

        np.savez(direc+'/V='+str(V)+'.npz', **save_dict)


