import numpy as np
from matplotlib import pyplot as plt
import math 
import time


#method1 escaping from the saddle point using perturbation
class saddlepoint:
    def __init__(self,x0=None,l=1,row=1,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[-0.005,1,1],max_itr=500):
        self.eigenvalue=eigenvalue
        self.eta=1/l
        self.max_itr=max_itr
        self.epsilon=epsilon
        self.d=len(eigenvalue)
        self.grad=np.zeros([self.d,1])
        if not x0.any():
            self.x=np.random.randn(self.d).reshape(self.d,1)
        else :
            self.x=x0
        self.x0=self.x
        self.x_tnoise=self.x
        self.hessian=np.diag(eigenvalue)
        self.prediction=self.obj_func(self.x)
        self.pred_seq=[]
        
        #in this case Delta can be computed directly   
        Delta=self.obj_func(self.x)


        self.chisq=3*max(np.log(self.d*l*Delta/(c*epsilon**2*delta)),4)
        self.eta=c/l
        self.r=c**(1/2)*epsilon/self.chisq**2/l*10000
        self.g_thres=c**(1/2)*epsilon/self.chisq**2*4000
        self.f_thres=c/self.chisq**3*(epsilon**3/row)**(1/2)*1000000
        self.t_thres=math.floor(self.chisq/c**2*l/(row*epsilon)**(1/2))
        self.t_noise=-self.t_thres-1

    def obj_func(self,x):
        return np.sum(np.dot(np.dot(x.T,self.hessian),x))
    def gradient(self,x):
        self.grad=np.dot(2*self.hessian,x)
    def fit(self,method='GD',verbose=True):
        assert method in ['SGD','GD','PGD']
        if method=='SGD':
            return self._sgd(verbose)
        elif method == 'GD':
            return self._gd(verbose)
        elif method== 'PGD':
            return self._pgd(verbose)
    def _gd(self,verbose=True):
        self.grad+=1
        self.x=self.x0
        self.pred_seq=[]
        i=0
        while np.linalg.norm(self.grad,ord=2)>self.epsilon*1 and i<self.max_itr:
            self.gradient(self.x)
            self.x=self.x-self.eta*self.grad
            self.prediction=self.obj_func(self.x)
            self.pred_seq.append(self.prediction)
            i=i+1
            if verbose:
                #print("iteration:{0},prediction={1:04f}".format(i,self.prediction))
                print("gradient:%1.04f,iteration:%d,prediction:%1.04f"%(np.linalg.norm(self.grad,ord=2),i,self.prediction))
    def _pgd(self,verbose=True):
        self.grad+=1
        self.x=self.x0
        self.pred_seq=[]
        t=0
        while True:
            if np.linalg.norm(self.grad,ord=2)<=self.g_thres and t-self.t_noise>self.t_thres:
                self.t_noise=t
                self.x_tnoise=self.x
                print('Before perturbabtion')
                print('x[0]:%1.04f,x[1]:%1.04f,x[2]:%1.04f'%(self.x[0][0],self.x[1][0],self.x[2][0]))
                self.__perturbation(self.r)
                print('After perturbabtion')
                print('x[0]:%1.04f,x[1]:%1.04f,x[2]:%1.04f'%(self.x[0][0],self.x[1][0],self.x[2][0]))

            if t-self.t_noise==self.t_thres and self.obj_func(self.x)-self.obj_func(self.x_tnoise)> -self.f_thres:
                break;
            elif t>self.max_itr :
                break;
            else :
                self.gradient(self.x)
                self.x=self.x-self.eta*self.grad
                self.prediction=self.obj_func(self.x)
                self.pred_seq.append(self.prediction)
            t=t+1
            if verbose:
                print("gradient:%1.04f,iteration:%d,prediction:%1.04f"%(np.linalg.norm(self.grad,ord=2),t,self.prediction))


    def __perturbation(self,r):
        rand=np.random.uniform(0,2*math.pi,size=self.d-1)
        for i in range(self.d):
            if i ==0:
                self.x[i][0]+=self.__multi_cos(rand[0:self.d-i-1])*r
            else :
                self.x[i][0]+=self.__multi_cos(rand[0:self.d-i-1])*math.sin(rand[self.d-i-1])*r
            
    def __multi_cos(self,a):
        if not a.any():
            return 1
        else:
            result=1
            for i in range(len(a)):
                result=result*math.cos(a[i])
        return result
    

#test for method_escaping from the saddle point by random permutation   
#x0=np.array([0,2,3])
time_start=time.time()
x0=np.random.uniform(1,6,3)
result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[-0.01,0.85,0.85],max_itr=1000)
result.fit('GD',verbose=False)
plt.ylabel('objective function ')
plt.xlabel('test number')
plt.plot(result.pred_seq)
time_end=time.time()
print('totally cost',time_end-time_start)





'''
t=[]
for i in range(300):
    eps=0.0001+0.00001*i
    time_start=time.time()
    result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=eps, c=1, delta=0.1,Delta=3,eigenvalue=[0.61,0.85,0.85],max_itr=10000)
    result.fit('PGD',verbose=False)
    plt.plot(result.pred_seq)
    time_end=time.time()
    t.append((time_end-time_start)*eps**2/np.log(3*1*3/eps**2/0.1)**4)
    
t=[]
for d in range(500):
    eigen=np.random.randint(2,8,d)
    time_start=time.time()
    result=saddlepoint(x0=x0.reshape(3,1),l=1,row=10,epsilon=0.01, c=1, delta=0.1,Delta=3,eigenvalue=[0.61,0.85,0.85],max_itr=10000)
    result.fit('PGD',verbose=False)
    plt.plot(result.pred_seq)
    time_end=time.time()
    t.append((time_end-time_start)/np.log(d*3/0.01**2/0.1)**4)
    
    
t   
plt.ylabel('t/log(d*l*Delta_f/delta/eps^2)^4')
plt.xlabel('test numbe')
plt.plot(t[120:500])
plt.show()    
    
    
result.fit('GD',verbose=False)
plt.plot(result.pred_seq)
#in this case we can see that random permutation did help escaping from the saddle point.
'''