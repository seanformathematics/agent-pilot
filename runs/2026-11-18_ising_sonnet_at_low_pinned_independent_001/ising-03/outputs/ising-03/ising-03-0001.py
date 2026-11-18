
import random, math, time, json
import numpy as np
N=16;L=N*N
nb=[[((i//N+1)%N)*N+i%N,((i//N-1)%N)*N+i%N,(i//N)*N+(i+1)%N,(i//N)*N+(i-1)%N] for i in range(L)]
def chain(T,seed,nmeas,neq):
    rnd=random.Random(seed);p=1-math.exp(-2/T)
    s=[rnd.choice((-1,1)) for _ in range(L)];M=sum(s)
    m2=0.;m4=0.
    for it in range(neq+nmeas):
        i=rnd.randrange(L);c=s[i];s[i]=-c;st=[i];n=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==c and rnd.random()<p:
                    s[k]=-c;st.append(k);n+=1
        M+=-2*c*n
        if it>=neq:
            x=(M/L)**2;m2+=x;m4+=x*x
    return m2/nmeas,m4/nmeas
res={}
t0=time.time()
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    Us=[]
    for c in range(10):
        a,b=chain(T,1000*int(T*10)+c,4000,500)
        Us.append((a,b))
    a=np.mean([u[0] for u in Us]);b=np.mean([u[1] for u in Us])
    U=1-b/(3*a*a)
    # jackknife over chains
    jk=[]
    for k in range(10):
        o=[u for q,u in enumerate(Us) if q!=k]
        aa=np.mean([u[0] for u in o]);bb=np.mean([u[1] for u in o]);jk.append(1-bb/(3*aa*aa))
    err=math.sqrt(9/10*sum((x-np.mean(jk))**2 for x in jk))
    res[T]=(U,err);print(T,U,err,time.time()-t0,flush=True)
json.dump(res,open('/work/r.json','w'))
