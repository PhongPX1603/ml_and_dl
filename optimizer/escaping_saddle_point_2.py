#Another method-- Natasha 

#part1 Natasha 1.5
import pandas as pd
class Natasha:
    def __init__(self,n,epsilon,sigma,L,L2,alpha,eigenvalue,v,ddelta,Delta_f,x0):
        self.n=n
        self.L=L
        self.L2=L2
        self.ddelta=ddelta
        self.p0=len(x0)
        self.p=math.floor((sigma/epsilon/L)**(2/3))
        self.B=math.floor(1/epsilon**2*0.001)
        self.m=math.floor(self.B/self.p*10)
        self.T=L**(2/3)*sigma**(1/3)/epsilon**(10/3)
        self.T2=math.floor(self.T/self.B*0.001)
        self.obj_eig_value=self.init_eigen(eigenvalue)
        self.x0=x0.reshape([self.p0,1])
        self.x1=self.x0
        self.x2=self.x0
        self.y=self.x0
        self.mu=np.zeros([self.p0,1])
        self.X=[]
        self.delta=np.zeros([self.p0,1])
        self.sigma=sigma
        self.alpha=alpha
        self.sigma_til=L2*v**(1/3)*epsilon**(1/3)/ddelta
        self.prediction=[]
        if L*ddelta/v**(1/3)/epsilon**(1/3):
            self.L_til=self.sigma_til
        else :
            self.L_til=L
            self.sigma_til=max(v*epsilon*L2**(3)/L**2/ddelta**(3),epsilon*L/v**(1/2))
        
        self.N1=self.sigma_til*Delta_f/self.p/epsilon**2*0.01
        self.Y=[]
        self.y_k=self.x0
        
            
        
    def objective_func(self,x,eigenvalue)  :
        return np.sum(np.dot(np.dot(x.T,np.diag(eigenvalue)),x))
    def init_eigen(self,eigenvalue):
        result=np.zeros([self.n,len(eigenvalue)])
        for i in range(self.n):
            result[i]=eigenvalue+np.random.randn(len(eigenvalue))*0.05*self.L
        return result
    def gradient(self,x,eigenvalue):
        if np.linalg.norm(x-self.y_k)<=self.ddelta/self.L2:
            return np.dot(2*np.diag(eigenvalue),x.reshape([self.p0,1]))
        else :
            return  np.dot(2*np.diag(eigenvalue),x.reshape([self.p0,1]))+2*self.L*(np.linalg.norm(x-self.y_k)-self.ddelta/self.L2)*(x-self.y_k)/np.linalg.norm(x-self.y_k)
        
        
    def Natasha1_5(self,x0):
        self.X=[]
        self.x0=x0
        self.x1=x0
        self.x2=x0
        for k in range(self.T2):
            self.x2=self.x1
            self.mu_renew()
            for s in range(self.p):
                x_choose=np.zeros([self.p0,self.m+1])
                x_choose[:,0]=self.x1.reshape([1,self.p0])
                self.X.append(self.x1)
                for t in range(self.m):
                    self.delta_renew(x_choose[:,t].reshape([self.p0,1]))
                    x_choose[:,t+1]=x_choose[:,t]-self.alpha*self.delta.reshape([1,self.p0])
                self.x1=x_choose.mean(1).reshape(self.p0,1)
            
        #self.y=self.mean_list(self.X)    
        self.y=self.X[-1]
        x_output=self.sgd(self.y,self.alpha,100)
        return x_output
          
    def mu_renew(self):
        result=np.zeros([self.p0,1])
        for i in range(self.B):
            result=result+self.gradient(self.x2,self.obj_eig_value[np.random.choice(self.n)])
        result=result/self.B
        self.mu=result
        
    def delta_renew(self,x):
        i=np.random.choice(self.n)
        self.delta=self.gradient(x,self.obj_eig_value[i])-self.gradient(self.x2,self.obj_eig_value[i])+self.mu.reshape([self.p0,1])+2*self.sigma*((x-self.x1))
        
    def mean_list(self,X):
        return sum(X)/len(X)
    
    def sgd(self,y,eta,maxitr):
        x_iter=y
        for j in range(maxitr):
            i=np.random.choice(self.n)
            delta=self.gradient(x_iter,self.obj_eig_value[i])+2*self.sigma*(x_iter-y)
            x_iter=x_iter-eta*delta
        return x_iter
        
    
    #Oja_algorithm 
    #input the basic eta,p,L,delta,d,C and output [judge,v]
    #if judge ==yes means we found the vector v so that we can move in the direction of v
    #if judge ==False means the minimum eigenvalue is above the threshold, so wecan go into the first order step
    def Oja_alg(self,eta,p,L,delta,d,C):
        T1=math.floor(np.log(1/p))
        T2=math.floor(12**2*C**2*L**2/delta**2*(np.log(d/p))**2)
        s=[]
        vector_s=[]
        for k in range(T1):
            W=[]
            a=np.random.uniform(0,1,d)
            W.append(a/(sum(a**2))**(1/2))
            sum_eigen=0
            for i in range(T2-1):
                #从n里面抽样
                eigen=self.obj_eig_value[np.random.choice(self.n)]
                #迭代得到下一个w
                times=np.dot((np.identity(d)+eta*  (0.5*np.identity(d)-np.diag(eigen)/2/L   )    ),W[i])
                W.append(times/sum(times**2)**(1/2))
                sum_eigen=sum_eigen+eigen
            sum_eigen=sum_eigen/T2
            #从0...T2-1中随机抽取一个W_i
            i_rand=np.random.choice(T2)   
            #计算得到s
            s.append(np.dot(np.dot(W[i_rand],(0.5*np.identity(d)-np.diag(sum_eigen)/2/L) ),W[i_rand].reshape([d,1])))
            vector_s.append(W[i_rand])
        smin=max(s)
        row=L-2*L*smin
        v=vector_s[s.index(smin)]
        if row>=-4*C*2*L*np.log(d/p)/T2**(1/2):
            judge=False
        else :
            judge=True
        return [judge,v]    

    
    
    def Natasha2(self,y0,eps,delta):
        count=0
        count2=0
        while(True):
            result=self.Oja_alg(eta=0.5,p=0.0001,L=1,delta=delta,d=3,C=2*10**(-3))
            if result[0]==True:
                self.y_k=self.y_k+(np.random.choice(2)*2-1)*self.ddelta/self.L2*result[1].reshape([self.p0,1])
            else:
                self.y_k=self.Natasha1_5(self.y_k)
                count=count+1
                self.Y.append(self.y_k)
            count2=count2+1
            self.prediction.append(self.objective_func(self.y_k,np.sum(self.obj_eig_value,axis=0)/self.n ))
            if count>=self.N1 or count2>100:
                break
        return self.y_k
    
        
            
        
        
 
result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[-0.05,0.85,0.85],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
result.Natasha1_5(result.x0)
result.Oja_alg(eta=0.5,p=0.0001,L=1,delta=0.05,d=3,C=2*10**(-3))
result.Natasha2(result.x0,eps=0.01,delta=0.05)









     
#test for Natasha1.5

t=[]
for i in range(50):
    L=1+0.01*i
    result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[-0.05,0.85,0.85],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
    t_start=time.time()
    result.Natasha1_5(result.x0)
    t_end=time.time()
    t.append((t_end-t_start)/L**(2/3))
    
    
plt.ylabel('t*eps^(3.25)')
plt.xlabel('test number')
plt.plot(t)
plt.show()


 


#test for Natasha2

t=[]
for i in range(50):
    eps=0.01+0.001*i
    result=Natasha(n=50,epsilon=0.01,sigma=1,L=1,L2=1,alpha=0.01,eigenvalue=[0.2,0.8,0.8],v=1,ddelta=1,Delta_f=10,x0=np.array([1,2,3]))
    #result.Natasha1_5(result.x0)
    #result.Oja_alg(eta=0.5,p=0.0001,L=1,delta=0.05,d=3,C=2*10**(-3))
    t_start=time.time()
    result.Natasha2(result.x0,eps=0.01,delta=1)
    t_end=time.time()
    t.append((t_end-t_start)*eps**(3.25))
    
    
plt.ylabel('t/L^(2/3)')
plt.xlabel('test number')
plt.plot(t)
plt.show()
