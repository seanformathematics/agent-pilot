
import random, math, time, json
import numpy as np
N=16;L=N*N
nb=[[((i//N)*N+(i%N+1)%N),((i//N)*N+(i%N-1)%N),(((i//N+1)%N)*N+i%N),(((i//N-1)%N)*N+i%N)] for i in range(L)]
def run(T,secs,seed):
    rnd=random.Random(seed).random
    p=1-math.exp(-2/T)
    s=[1]*L if T<2.3 else [random.Random(seed+1).choice([-1,1]) for _ in range(L)]
    M=sum(s);m2=[];m4=[]
    t0=time.time();k=0
    while True:
        k+=1
        i=int(rnd()*L);c=s[i];s[i]=-c;st=[i];cs=1
        while st:
            j=st.pop()
            for n in nb[j]:
                if s[n]==c and rnd()<p:
                    s[n]=-c;st.append(n);cs+=1
        M+=-2*c*cs
        if k>500:
            x=(M/L)**2;m2.append(x);m4.append(x*x)
        if k%500==0 and time.time()-t0>secs:break
    a=np.array(m2);b=np.array(m4)
    nbk=50;n=len(a)//nbk*nbk
    A=a[:n].reshape(nbk,-1).mean(1);B=b[:n].reshape(nbk,-1).mean(1)
    U=1-B.mean()/(3*A.mean()**2)
    jk=[1-np.delete(B,q).mean()/(3*np.delete(A,q).mean()**2) for q in range(nbk)]
    err=math.sqrt((nbk-1)/nbk*sum((u-np.mean(jk))**2 for u in jk))
    return U,err,n
res={}
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    res[T]=run(T,38,int(T*100));print(T,res[T],flush=True)
json.dump(res,open('/work/r.json','w'))
