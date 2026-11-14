
import numpy as np, random, math, time, json
N=16;L=N*N
nb=[[((i//N)*N+(i%N+1)%N),((i//N)*N+(i%N-1)%N),(((i//N+1)%N)*N+i%N),(((i//N-1)%N)*N+i%N)] for i in range(L)]
def chain(T,seed,nmeas,ntherm=500):
    rnd=random.Random(seed); p=1-math.exp(-2/T)
    s=[rnd.choice((-1,1)) for _ in range(L)]; M=sum(s)
    m2=[];m4=[]
    for it in range(ntherm+nmeas):
        i=rnd.randrange(L); sp=s[i]; s[i]=-sp; st=[i]; c=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rnd.random()<p:
                    s[k]=-sp; st.append(k); c+=1
        M+=-2*sp*c
        if it>=ntherm:
            m=M/L; m2.append(m*m); m4.append(m**4)
    return np.mean(m2),np.mean(m4)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={};t0=time.time()
K=10;n=20000
for T in Ts:
    Us=[]
    for k in range(K):
        a,b=chain(T,1000*k+int(T*10),n); Us.append(1-b/(3*a*a))
    res[T]=(np.mean(Us),np.std(Us,ddof=1)/math.sqrt(K))
    print(T,res[T],time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
