
import numpy as np, random, math, time, json
N=16;L=N*N
nb=[[((i//N)*N+(i%N+1)%N),((i//N)*N+(i%N-1)%N),(((i//N+1)%N)*N+i%N),(((i//N-1)%N)*N+i%N)] for i in range(L)]
def chain(T,seed,nmeas,ntherm=500):
    rnd=random.Random(seed); p=1-math.exp(-2/T)
    s=[1]*L if rnd.random()<.5 else [-1]*L
    M=sum(s); m2=[];m4=[]
    for it in range(ntherm+nmeas):
        i=rnd.randrange(L); c=s[i]; s[i]=-c; st=[i]; n=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==c and rnd.random()<p:
                    s[k]=-c; st.append(k); n+=1
        M-=2*c*n
        if it>=ntherm:
            m=M/L; m2.append(m*m); m4.append(m**4)
    return np.mean(m2),np.mean(m4)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
for T in Ts:
    nch=20; nm=int(__import__('sys').argv[1]) if False else 3000
    a=np.array([chain(T,1000+k+int(T*100)*100,nm) for k in range(nch)])
    U=1-a[:,1].mean()/(3*a[:,0].mean()**2)
    Uk=1-a[:,1]/(3*a[:,0]**2)
    # jackknife
    jk=[1-np.delete(a[:,1],k).mean()/(3*np.delete(a[:,0],k).mean()**2) for k in range(nch)]
    err=math.sqrt((nch-1)/nch*np.sum((np.array(jk)-np.mean(jk))**2))
    res[T]=(U,err,nch*nm); print(T,U,err,time.time()-t0,flush=True)
json.dump(res,open('/work/res.json','w'))
