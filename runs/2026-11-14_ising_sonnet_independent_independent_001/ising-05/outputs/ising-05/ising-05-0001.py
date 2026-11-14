
import numpy as np, random, math, time, json
N=16;Ns=N*N
nb=[[((i//N)*N+(i%N+1)%N),((i//N)*N+(i%N-1)%N),(((i//N+1)%N)*N+i%N),(((i//N-1)%N)*N+i%N)] for i in range(Ns)]
def chain(T,seed,nmeas,neq):
    rnd=random.Random(seed); p=1-math.exp(-2/T)
    s=[rnd.choice((-1,1)) for _ in range(Ns)]
    M=sum(s); m2=m4=0.0
    for it in range(neq+nmeas):
        i=rnd.randrange(Ns); c=s[i]; s[i]=-c; st=[i]; n=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==c and rnd.random()<p:
                    s[k]=-c; st.append(k); n+=1
        M+=-2*c*n
        if it>=neq:
            m=M/Ns; m2+=m*m; m4+=m**4
    return m2/nmeas,m4/nmeas
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
for T in Ts:
    Us=[]
    nm=6000 if T<2.0 else 20000
    for c in range(10):
        a,b=chain(T,1000*int(T*10)+c,nm,500)
        Us.append(1-b/(3*a*a))
    Us=np.array(Us); res[T]=(Us.mean(),Us.std(ddof=1)/np.sqrt(10),10*nm)
    print(T,res[T],time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/r.json','w'))
