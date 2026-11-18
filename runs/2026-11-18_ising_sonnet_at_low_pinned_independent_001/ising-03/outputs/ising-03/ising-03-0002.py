
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
K=40
for T in [2.3,2.4,2.5,2.6,2.7,2.8]:
    Us=[chain(T,777000+int(T*10)*100+c,10000,500) for c in range(K)]
    A=np.array(Us);a,b=A.mean(0)
    U=1-b/(3*a*a)
    jk=[]
    for k in range(K):
        o=np.delete(A,k,0).mean(0);jk.append(1-o[1]/(3*o[0]**2))
    err=math.sqrt((K-1)/K*sum((x-np.mean(jk))**2 for x in jk))
    res[T]=(U,err);print(T,U,err,flush=True)
json.dump(res,open('/work/r2.json','w'))
