
import random, math, time, json
import numpy as np
N=16;NS=N*N
nb=[[((i//N)*N+(i%N+1)%N),((i//N)*N+(i%N-1)%N),(((i//N+1)%N)*N+i%N),(((i//N-1)%N)*N+i%N)] for i in range(NS)]
def chain(T,seed,neq,nm):
    rnd=random.Random(seed);rr=rnd.random
    p=1-math.exp(-2/T)
    s=[rnd.choice((-1,1)) for _ in range(NS)]
    M=sum(s);m2=[];m4=[]
    for it in range(neq+nm):
        i=int(rr()*NS);sp=s[i];s[i]=-sp;st=[i];c=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rr()<p:
                    s[k]=-sp;st.append(k);c+=1
        M-=2*sp*c
        if it>=neq:
            m=M/NS;m2.append(m*m);m4.append(m**4)
    return np.mean(m2),np.mean(m4)
Ts=[2.4,2.5,2.6,2.7,2.8]
res={};t0=time.time()
for T in Ts:
    nc=10;nm=200000
    A=[];B=[]
    for c in range(nc):
        a,b=chain(T,50000+int(T*10)*100+c,1000,nm);A.append(a);B.append(b)
    U=1-np.mean(B)/(3*np.mean(A)**2)
    jk=[1-np.mean(np.delete(B,c))/(3*np.mean(np.delete(A,c))**2) for c in range(nc)]
    err=math.sqrt((nc-1)/nc*np.sum((np.array(jk)-np.mean(jk))**2))
    res[T]=(U,err,nc*nm);print(T,U,err,time.time()-t0,flush=True)
