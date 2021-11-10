import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys

import imageio
from PIL import Image

class Beads:     # OOP
    """basic model to simulate particle cluster of active particles under confinement. d1 : ensemble, d2 : particles, d3 : subparticles"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,L, N_ptcl,N_active,N_sub,AR,r_b,N_ensemble,Fs,g):
        
        
        # set up coefficients
        self.set_coeff(L,N_ptcl,N_active,N_ensemble,Fs,g) 
        self.set_ellipse(N_sub,AR,r_b)
      
        # initializing configuration of state
        self.X = np.ones(self.N_ensemble).reshape(-1,1,1)*np.linspace(0,self.L,self.N_ptcl+1)[:self.N_ptcl].reshape(1,-1,1)
        self.O = np.ones((self.N_ensemble,self.N_active,1))*np.pi/2
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
        self.D = 5000
        
        
        # dynamics
        self.p = 2000    # propulsion
        self.mu = 0.002
        self.mur = 0.002
        
        
        
        
        self.g = g
        
        
        
        self.Omin = 0
        
    def set_ellipse(self,N_sub,AR,r_b):
        self.N_sub = N_sub
        self.r_0 =  1 # radius of passive bead
        self.r_b = r_b
        self.AR = AR
        
        
        # inner structure coefficients
        self.k = 1         # epsilon of WCA potential
#         self.k2 = 0.1
        
        
        
        theta = np.linspace(0,np.pi,self.N_sub,endpoint=False) 
        self.l = self.r_b*(AR-1/AR)*(AR+np.cos(theta)).reshape(1,1,-1)     # center position of sub bead
        self.r_sub = self.AR*self.r_b*np.sqrt(1+(1/AR**2-1)*np.cos(theta)**2).reshape(1,1,-1)  # radius of active sub beads r1,r2, ...
    
    
    
    
    
    # boundary condition
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        
        return mod_x
        
        
        
    # Dynamics part
    def set_zero(self,flag):              # initializing simulation configurations
        
#         self.X = np.ones(self.N_ensemble).reshape(-1,1)*np.linspace(0,self.L,self.N_ptcl+1)[:self.N_ptcl].reshape(1,-1)
        self.X[flag]= np.zeros((np.sum(flag),self.N_ptcl,1))
        
        l1 = self.r_0*(self.N_ptcl-self.N_active)
        l2 = self.r_b*(self.N_active)
        self.X[flag,1:self.N_active+1] = np.ones(np.sum(flag)).reshape(-1,1,1)*np.linspace(0,self.L*(l2/(l1+l2)),self.N_active+1)[1:].reshape(1,-1,1)# active ones
        self.X[flag,self.N_active+1:] = np.ones(np.sum(flag)).reshape(-1,1,1)*np.linspace(self.L*(l2/(l1+l2)),self.L,self.N_ptcl-self.N_active+1)[1:-1].reshape(1,-1,1)     # passive ones
        
        self.O[flag] = np.ones((np.sum(flag),self.N_active,1))*np.pi/2
        
        self.set_structure()
        
    def set_structure(self):
        self.Xs = self.X[:,1:self.N_active+1].reshape((self.N_ensemble,self.N_active,1))+self.l*np.cos(self.O.reshape((self.N_ensemble,self.N_active,1)))   # 1~N_active
        self.Ys = self.l*np.sin(self.O.reshape((self.N_ensemble,self.N_active,1)))
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
#         return force*(np.divide(rx,r,out=np.zeros_like(rx),where=r!=0),np.divide(ry,r,out=np.zeros_like(ry),where=r!=0))
        return force*(rx/r,ry/r)
    
    def force(self):    # force from WCA potential (truncated LJ potential -> hard wall repulsion)
        f1x = np.zeros((self.N_ensemble,self.N_ptcl,1))
        f2x = np.zeros((self.N_ensemble,self.N_active,1))
#         f1y = np.zeros(self.N_active)
#         f2y = np.zeros((self.N_ensemble,self.N_active))
        torque = np.zeros((self.N_ensemble,self.N_active,1))


        
        # particle 1 constrained to wall, particle 2 rotating
        
        # relative position from -> to : r(from)-r(to)
        
        # 1->1
        relx_right = self.periodic(np.roll(self.X,-1,axis=1)-self.X)
        relx_left = self.periodic(np.roll(self.X,1,axis=1)-self.X)
        rely_right = np.zeros((self.N_ensemble,self.N_ptcl,1))
        rely_left = np.zeros((self.N_ensemble,self.N_ptcl,1))
        rely_right[:,0] = -self.r_0+self.r_b
        rely_left[:,self.N_active+2] = self.r_0-self.r_b
        
        lengthR = np.concatenate(([self.r_0+self.r_b],(self.r_b*2)*np.ones(self.N_active-1),[self.r_0+self.r_b],(self.r_0*2)*np.ones(self.N_ptcl-self.N_active-1))).reshape(1,-1,1)
        lengthL = np.concatenate(([self.r_0*2],[self.r_0+self.r_b],(self.r_b*2)*np.ones(self.N_active-1),[self.r_0+self.r_b],(self.r_0*2)*np.ones(self.N_ptcl-self.N_active-2))).reshape(1,-1,1)

        (fx,fy)=self.WCA(relx_right,rely_right,lengthR)
        f1x += fx # right particle
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL)
        f1x += fx  # left particle
        
        
        # 1->2
#         print(self.Xs.shape)
        relx_right = self.periodic(self.X[:,2:self.N_active+2]-self.Xs)
        relx_left = self.periodic(self.X[:,:self.N_active]-self.Xs)
        rely_right = -self.Ys
        rely_left = -self.Ys
        rely_right[:,-1] += self.r_0-self.r_b
        rely_left[:,0] -= self.r_0-self.r_b
        
        lengthR = np.concatenate(((self.r_b+self.r_sub)*np.ones((1,self.N_active-1,1)),(self.r_0+self.r_sub)),axis=1)
        lengthL = np.concatenate(((self.r_0+self.r_sub),(self.r_b+self.r_sub)*np.ones((1,self.N_active-1,1))),axis=1)
        
        (fx,fy)=self.WCA(relx_right,rely_right,lengthR)
        f2x+=np.sum(fx,axis=2)[:,:,np.newaxis]  
        torque+=np.sum(-fx*np.sin(self.O)+fy*np.cos(self.O),axis=2)[:,:,np.newaxis]  
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL)
        f2x+=np.sum(fx,axis=2)[:,:,np.newaxis]  
        torque+=np.sum(-fx*self.l*np.sin(self.O)+fy*self.l*np.cos(self.O),axis=2)[:,:,np.newaxis]  
        
        
        # 2->1
        
        relx_right = self.periodic(self.Xs-self.X[:,:self.N_active])
        relx_left = self.periodic(self.Xs-self.X[:,2:self.N_active+2])
        rely_right = self.Ys
        rely_left = self.Ys
        rely_right[:,0] -= self.r_0-self.r_b
        rely_left[:,-1] += self.r_0-self.r_b
        
        lengthR = np.concatenate(((self.r_0+self.r_sub),(self.r_b+self.r_sub)*np.ones((1,self.N_active-1,1))),axis=1)
        lengthL = np.concatenate(((self.r_b+self.r_sub)*np.ones((1,self.N_active-1,1)),(self.r_0+self.r_sub)),axis=1)
        
        (fx,fy)=self.WCA(relx_right,rely_right,lengthR)
        f1x[:,:self.N_active]+=np.sum(fx,axis=2)[:,:,np.newaxis]  
        (fx,fy)=self.WCA(relx_left,rely_left,lengthL)
        f1x[:,2:self.N_active+2]+=np.sum(fx,axis=2)[:,:,np.newaxis]  
        
        
        # 2->2
        for i in range(self.N_sub):
            relx_right = self.periodic(self.Xs[:,1:,i][:,:,np.newaxis] -self.Xs[:,:-1])
            relx_left = self.periodic(self.Xs[:,:-1,i][:,:,np.newaxis]  -self.Xs[:,1:])
            rely_right = self.Ys[:,1:,i][:,:,np.newaxis]  -self.Ys[:,:-1]
            rely_left = self.Ys[:,:-1,i][:,:,np.newaxis]  -self.Ys[:,1:]

            (fx,fy)=self.WCA(relx_right,rely_right,self.r_sub[0,0,i]+self.r_sub)
            f2x[:,:-1]+=np.sum(fx,axis=2)[:,:,np.newaxis]    
            torque[:,:-1]+=np.sum(-fx*self.l*np.sin(self.O[:,:-1])+fy*self.l*np.cos(self.O[:,:-1]),axis=2)[:,:,np.newaxis]  
            (fx,fy)=self.WCA(relx_left,rely_left,self.r_sub[0,0,i]+self.r_sub)
            f2x[:,1:]+=np.sum(fx,axis=2)[:,:,np.newaxis]  
            torque[:,1:]+=np.sum(-fx*self.l*np.sin(self.O[:,1:])+fy*self.l*np.cos(self.O[:,1:]),axis=2)[:,:,np.newaxis]         
        
        # noise
        f1x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_ptcl))[:,:,np.newaxis]
#         f2x+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_active))
#         f2y+=np.random.normal(0,np.sqrt(2*self.D/self.dt),(self.N_ensemble,self.N_active))
        
        # normal force
#         fN = self.p*np.sin(self.O)-f1y

        # gravity
#         f2y-=self.g
#         torque-=self.g*np.cos(self.O)
        
        return(f1x,f2x,torque)
        

    
    def time_evolve(self):
        
        # compute force & torque
        (f1x,f2x,Torque) = self.force()
        Fx = f1x
        Fx[:,1:self.N_active+1]+=f2x-self.p*np.cos(self.O)
        
#         self.v = np.mean(Fx,axis=1)*self.mu
        self.v = np.sum(np.cos(self.O),axis=1)  #*self.mu

        
#         Torque = self.l/2*(f2y*np.cos(self.O)+(f1x[:, 1:self.N_active+1]-f2x-self.p*np.cos(self.O))*np.sin(self.O))
        # update configuration
        self.X+=self.mu*Fx*self.dt
        self.O+=self.mur*Torque*self.dt

        self.X = self.periodic(self.X)
        
        self.O = np.amax(np.vstack([[self.O],[np.ones((self.N_ensemble,self.N_active,1))*self.Omin]]),axis=0)
        self.O = np.amin(np.vstack([[self.O],[(np.ones((self.N_ensemble,self.N_active,1))*(np.pi-self.Omin))]]),axis=0)  

        self.set_structure()
        
    
        
    def animate(self,N_iter,directory):
        self.set_zero((np.ones(self.N_ensemble)==np.ones(self.N_ensemble)))
        
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
        Othres = self.Omin+np.pi/50
        
        
        for nn in trange(N_iter):
            
            ax1.clear()
            ax1.scatter(self.X[0][0],self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax1.scatter(self.X[0][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax1.scatter(self.X[0][1:self.N_active+1],np.zeros(self.N_active)+self.r_b,s=self.r_b**2*200000/self.L**2,color='green')

            for i in range(self.N_sub):
                ax1.scatter(self.Xs[0,:,i],self.Ys[0,:,i]+self.r_b,s=self.r_sub[0,0,i]**2*200000/self.L**2,color='blue')
            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')
            
            
            
            ax2.clear()
            ax2.scatter(self.X[1][0],self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax2.scatter(self.X[1][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax2.scatter(self.X[1][1:self.N_active+1],np.zeros(self.N_active)+self.r_b,s=self.r_b**2*200000/self.L**2,color='green')

            for i in range(self.N_sub):
                ax2.scatter(self.Xs[1,:,i],self.Ys[1,:,i]+self.r_b,s=self.r_sub[0,0,i]**2*200000/self.L**2,color='blue')
            ax2.axis(axrange)
            ax2.set_aspect('equal', 'box')
            
            
            
            ax3.clear()
            ax3.scatter(self.X[2][0],self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax3.scatter(self.X[2][self.N_active+1:],np.zeros(self.N_ptcl-self.N_active-1)+self.r_0,s=self.r_0**2*200000/self.L**2,color='red')
            ax3.scatter(self.X[2][1:self.N_active+1],np.zeros(self.N_active)+self.r_b,s=self.r_b**2*200000/self.L**2,color='green')

            for i in range(self.N_sub):
                ax3.scatter(self.Xs[2,:,i],self.Ys[2,:,i]+self.r_b,s=self.r_sub[0,0,i]**2*200000/self.L**2,color='blue')
            ax3.axis(axrange)
            ax3.set_aspect('equal', 'box')
            
            
            
            
            fig1.canvas.draw()
            if True:
                fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
            stuck = (np.cos(self.O[:,0,0])<-np.cos(Othres))*(np.cos(self.O[:,-1,0])>np.cos(Othres))
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
        
        prev_right = (np.cos(self.O[:,0])<-np.cos(Othres))*(~(np.cos(self.O[:,-1])>np.cos(Othres)))
        prev_left = (np.cos(self.O[:,-1])>np.cos(Othres))*(~(np.cos(self.O[:,0])<-np.cos(Othres)))
        prev_stuck = (np.cos(self.O[:,0])<-np.cos(Othres))*(np.cos(self.O[:,-1])>np.cos(Othres))
        time = 0

        v1_t = np.zeros(N_iter)
        v2_t = np.zeros(N_iter)
        for j in trange(N_iter):
#          
            
    
            self.time_evolve()
            
            right = (np.cos(self.O[:,0])<-np.cos(Othres))*(~(np.cos(self.O[:,-1])>np.cos(Othres)))
            left = (np.cos(self.O[:,-1])>np.cos(Othres))*(~(np.cos(self.O[:,0])<-np.cos(Othres)))
            stuck = (np.cos(self.O[:,0,0])<-np.cos(Othres))*(np.cos(self.O[:,-1,0])>np.cos(Othres))
        
            time = j*self.dt#*N_time
            
    
    
                    
            for i in range(self.N_ensemble):
                if right[i]:
                    if(not prev_right[i]):
                        move_in[i] = np.append(move_in[i],time)
                        age[i] = 0
                        count+=1
                    v1_t[age[i]]+=np.abs(self.v[i])
                    v2_t[age[i]]+=self.v[i]**2
                elif left[i]:
                    if (not prev_left[i]):
                        move_in[i] = np.append(move_in[i],time)
                        age[i] = 0   
                        count+=1
                    v1_t[age[i]]+=np.abs(self.v[i])
                    v2_t[age[i]]+=self.v[i]**2
                elif stuck[i]*(prev_right[i] or prev_left[i]):
                    move_out[i] = np.append(move_out[i],time)
            self.set_zero(stuck)
            age +=1
            
                    
            prev_right = right
            prev_left = left
            prev_stuck = stuck
            
            
        time +=self.dt
        for i in range(self.N_ensemble):
            if right[i]:
                move_out[i] = np.append(move_out[i],time)
            elif left[i]:
                move_out[i] = np.append(move_out[i],time)
        
        
        v_t_avg = v1_t/count
        v_t_var = v2_t/count-(v1_t/count)**2
        
        
#             elif stuck[i]:
#                 stuck_out[i] = np.append(stuck_out[i],time)

#         return(right_in,left_in,stuck_in, right_out, left_out,stuck_out)

        return(move_in, move_out,v_t_avg,v_t_var)


          
            
        
def time(N_ptcl, N_active,g,D):

    B1 = Beads(L=68, N_ptcl = N_ptcl,N_active = N_active,N_sub = 5,AR=1.5,r_b = 1,N_ensemble = 100,Fs=3000,g=g)

    B1.p = 200
    B1.D = D
    B1.mu = 0.1
    B1.mur = 0.2
    B1.Omin = 0
    B1.k = 0.1
    B1.r_0 =1.5
    # B1.r_cut = [1.3,0.8,0.9]


    B1.L = ((B1.N_ptcl-B1.N_active)*2*B1.r_0+(B1.N_active)*2*B1.r_b+2*(B1.AR-1)*B1.r_b)*0.95



    direc = '211110_v_t/N_ptcl='+str(B1.N_ptcl)+',g='+str(B1.g)+',D='+str(B1.D)
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


