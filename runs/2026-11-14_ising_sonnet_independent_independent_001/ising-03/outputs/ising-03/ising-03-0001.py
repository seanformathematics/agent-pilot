
import numpy as np, random, math, time, json
N=16;L=N*N
nb=[[((i//N+1)%N)*N+i%N,((i//N-1)%N)*N+i%N,(i//N)*N+(i+1)%N,(i//N)*N+(i-1)%N] for i in range(L)]
def run(T,seed,nth,nmeas):
    rnd=random.Random(seed);p=1-math.exp(-2/T)
    s=[1]*L if T<2.3 else [rnd.choice((-1,1)) for _ in range(L)]
    M=sum(s);m2=0.;m4=0.;rnd_=rnd.random
    for it in range(nth+nmeas):
        i=rnd.randrange(L);sp=s[i];s[i]=-sp;st=[i];c=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rnd_()<p:
                    s[k]=-sp;st.append(k);c+=1
        M-=2*sp*c
        if it>=nth:
            m=M/L;a=m*m;m2+=a;m4+=a*a
    return 1-(m4/nmeas)/(3*(m2/nmeas)**2)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={};t=time.time()
for T in Ts:
    us=[run(T,1000+int(T*10)*100+r,1000,20000) for r in range(10)]
    res[T]=(np.mean(us),np.std(us,ddof=1)/np.sqrt(10))
    print(T,res[T],time.time()-t,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/r.json','w'))
