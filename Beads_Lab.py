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
        
        
        # set up coefficients
        self.set_coeff(L,N_ptcl,N_active,N_ensemble,Fs,g) 
      
        # initializing configuration of state
        self.set_zero()
        
        
        
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
        self.D = 5000
        
        
        # dynamics
        self.p = 2000    # propulsion
        self.mu = 0.002
        self.mur = 0.002
        
        # inner structure coefficients
        self.k1 = 1         # epsilon of WCA potential
        self.k2 = 0.1
        self.l = 1    # length between fixed beads
        self.r_cut = [1.5,1.65,1.8]  # radius of beads [r1+r1,r1+r2,r2+r2]
        self.g = 1
        
        
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        
#         self.X = np.ones(self.N_ensemble).reshape(-1,1)*np.linspace(0,self.L,self.N_ptcl+1)[:self.N_ptcl].reshape(1,-1)
        self.X = np.zeros((self.N_ensemble,self.N_ptcl))
        
        l1 = self.r_cut[0]*(self.N_ptcl-self.N_active)
        l2 = self.r_cut[1]*(self.N_active)
        self.X[:,1:self.N_active+1] = np.ones(self.N_ensemble).reshape(-1,1)*np.linspace(0,self.L*(l2/(l1+l2)),self.N_active+1)[1:].reshape(1,-1)# active ones
        self.X[:,self.N_active+1:] = np.ones(self.N_ensemble).reshape(-1,1)*np.linspace(self.L*(l2/(l1+l2)),self.L,self.N_ptcl-self.N_active+1)[1:-1].reshape(1,-1)     # passive ones
        
        self.O = np.ones((self.N_ensemble,self.N_active))*np.pi/2
        
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
        relx_right = self.periodic(np.roll(self.X,-1,axis=1)-self.X)
        relx_left = self.periodic(np.roll(self.X,1,axis=1)-self.X)
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
        relx_right = self.periodic(self.X[:,2:self.N_active+2]-self.Xs)
        relx_left = self.periodic(self.X[:,:self.N_active]-self.Xs)
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
        
        relx_right = self.periodic(self.Xs-self.X[:,:self.N_active])
        relx_left = self.periodic(self.Xs-self.X[:,2:self.N_active+2])
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
        relx_right = self.periodic(self.Xs[:,1:]-self.Xs[:,:-1])
        relx_left = self.periodic(self.Xs[:,:-1]-self.Xs[:,1:])
        rely_right = self.Ys[:,1:]-self.Ys[:,:-1]
        rely_left = self.Ys[:,:-1]-self.Ys[:,1:]
        
        (fx,fy)=self.WCA(relx_right,rely_right,self.r_cut[2]*2,self.k2)
        f2x[:,:-1]+=fx
        f2y[:,:-1]+=fy
        (fx,fy)=self.WCA(relx_left,rely_left,self.r_cut[2]*2,self.k2)
        f2x[:,1:]+=fx
        f2y[:,1:]+=fy
        
        
        # noise
        f1x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_ptcl))
#         f2x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_active))
#         f2y+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_active))
        
        # normal force
#         fN = self.p*np.sin(self.O)-f1y

        # gravity
        f2y-=self.g
        
        return(f1x,f2x,f2y)
        

    
    def time_evolve(self):
        
        # compute force & torque
        (f1x,f2x,f2y) = self.force()
        Fx = f1x
        Fx[:,1:self.N_active+1]+=f2x-self.p*np.cos(self.O)
        
#         self.v = np.mean(Fx,axis=1)*self.mu
        self.v = np.sum(np.cos(self.O),axis=1)  #*self.mu

        
        Torque = self.l/2*(f2y*np.cos(self.O)+(f1x[:, 1:self.N_active+1]-f2x-self.p*np.cos(self.O))*np.sin(self.O))
        # update configuration
        self.X+=self.mu*Fx*self.dt
        self.O+=self.mur*Torque*self.dt

        self.X = self.periodic(self.X)
        Omin = np.arcsin((self.r_cut[2]-self.r_cut[1])/self.l)
        self.O = np.amax(np.vstack([[self.O],[np.ones((self.N_ensemble,self.N_active))*Omin]]),axis=0)
        self.O = np.amin(np.vstack([[self.O],[(np.ones((self.N_ensemble,self.N_active))*(np.pi-Omin))]]),axis=0)  


        self.set_structure()
        
    
        
    def animate(self,N_iter,directory):
        self.set_zero()
        
        axrange = [-self.L/2, self.L/2, -self.L/100, self.L/10]
        
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
        self.set_zero()
        right_in = [np.zeros(0)]*self.N_ensemble
        left_in = [np.zeros(0)]*self.N_ensemble
        stuck_in = [np.zeros(0)]*self.N_ensemble
        right_out = [np.zeros(0)]*self.N_ensemble
        left_out = [np.zeros(0)]*self.N_ensemble
        stuck_out = [np.zeros(0)]*self.N_ensemble

        self.time_evolve()
        
        Omin = np.arcsin((self.r_cut[2]-self.r_cut[1])/self.l)
        
        prev_right = (np.cos(self.O[:,0])<-np.cos(Omin+np.pi/10))*(~(np.cos(self.O[:,-1])>np.cos(Omin+np.pi/20)))
        prev_left = (np.cos(self.O[:,-1])>np.cos(Omin+np.pi/10))*(~(np.cos(self.O[:,0])<-np.cos(Omin+np.pi/20)))
        prev_stuck = (np.cos(self.O[:,0])<-np.cos(Omin+np.pi/10))*(np.cos(self.O[:,-1])>np.cos(Omin+np.pi/20))
        
#         N_time = 40,

        for j in trange(N_iter):
#             v_temp = self.v
#             for _ in range(N_time-1):
#                 self.time_evolve()
#                 v_temp += self.v
#             v_temp/=N_time
            
            
#             self.time_evolve()
#             right = (v_temp>0.5)
#             left = (v_temp<-0.5)
#             stuck = (-0.5<=v_temp)*(v_temp<=0.5)
            
    
            self.time_evolve()
            right = (np.cos(self.O[:,0])<-np.cos(Omin+np.pi/10))*(~(np.cos(self.O[:,-1])>np.cos(Omin+np.pi/20)))
            left = (np.cos(self.O[:,-1])>np.cos(Omin+np.pi/10))*(~(np.cos(self.O[:,0])<-np.cos(Omin+np.pi/20)))
            stuck = (np.cos(self.O[:,0])<-np.cos(Omin+np.pi/10))*(np.cos(self.O[:,-1])>np.cos(Omin+np.pi/20))
        
        
            time = j*self.dt  #*N_time
            
    
    
#             bool_ri = right&(~ prev_right)
#             right_in[bool_ri] = np.append(right_in[bool_ri],time*np.ones(np.sum(bool_ri)).reshape(-1,1),axis=1)
#             bool_li = left&(~ prev_left)
#             left_in[bool_li] = np.append(left_in[bool_li],time*np.ones(np.sum(bool_li)).reshape(-1,1),axis=1)
#             bool_si = stuck&(~ prev_stuck)
#             stuck_in[bool_si] = np.append(stuck_in[bool_si],time*np.ones(np.sum(bool_si)).reshape(-1,1),axis=1)
#             bool_ro = (~ right)&(prev_right)
#             right_out[bool_ro] = np.append(right_out[bool_ro],time*np.ones(np.sum(bool_ro)).reshape(-1,1),axis=1)
#             bool_lo = (~ left)&(prev_left)
#             left_out[bool_lo] = np.append(left_out[bool_lo],time*np.ones(np.sum(bool_lo)).reshape(-1,1),axis=1)
#             bool_so = (~ stuck)&(prev_stuck)
#             stuck_out[bool_so] = np.append(stuck_out[bool_so],time*np.ones(np.sum(bool_so)).reshape(-1,1),axis=1)
            
            for i in range(self.N_ensemble):
                if right[i]*(not prev_right[i]):
                    right_in[i] = np.append(right_in[i],time)
                elif left[i]*(not prev_left[i]):
                    left_in[i] = np.append(left_in[i],time)
                elif stuck[i]*(not prev_stuck[i]):
                    stuck_in[i] = np.append(stuck_in[i],time)
                    
                if (not right[i])*(prev_right[i]):
                    right_out[i] = np.append(right_out[i],time)
                elif (not left[i])*(prev_left[i]):
                    left_out[i] = np.append(left_out[i],time)
                elif (not stuck[i])*(prev_stuck[i]):
                    stuck_out[i] = np.append(stuck_out[i],time)
                    
            prev_right = right
            prev_left = left
            prev_stuck = stuck
        time +=self.dt
        for i in range(self.N_ensemble):
            if right[i]:
                right_out[i] = np.append(right_out[i],time)
            elif left[i]:
                left_out[i] = np.append(left_out[i],time)
            elif stuck[i]:
                stuck_out[i] = np.append(stuck_out[i],time)

        return(right_in,left_in,stuck_in, right_out, left_out,stuck_out)
            
            
            
        
def time(N_active):

    B1 = Beads(L=68, N_ptcl = 100,N_active = N_active,N_ensemble = 300,Fs=500,g=10)

    B1.p = 100
    B1.D = 50  #5
    B1.mu = 0.01
    B1.mur = 0.01
    B1.k1 = 20
    B1.k2 = 8
    B1.r_cut = [1.5,1.5,1.7]
    # B1.r_cut = [1.3,0.8,0.9]
    B1.l = 1.8


    B1.L = ((B1.N_ptcl-B1.N_active+1)*2*B1.r_cut[0]+(B1.N_active+1)*2*B1.r_cut[1])*0.95

    direc = '211022/g='+str(B1.g)
    os.makedirs(direc,exist_ok=True)



    (right_in,left_in,stuck_in, right_out, left_out,stuck_out) = B1.transit(2000000)

    right_in =np.array(right_in)
    left_in = np.array(left_in)
    stuck_in = np.array(stuck_in)
    right_out =np.array(right_out)
    left_out = np.array(left_out)
    stuck_out = np.array(stuck_out)

    save_dict={}
    save_dict['right_in'] = right_in
    save_dict['left_in'] = left_in
    save_dict['stuck_in'] = stuck_in
    save_dict['right_out'] = right_out
    save_dict['left_out'] = left_out
    save_dict['stuck_out'] = stuck_out
    
    np.savez(direc+'/N'+str(B1.N_active)+'.npz', **save_dict)


