
import random, math, time, numpy as np, json
N=16;NN=N*N
nb=[[((i//N+1)%N)*N+i%N,((i//N-1)%N)*N+i%N,(i//N)*N+(i%N+1)%N,(i//N)*N+(i%N-1)%N] for i in range(NN)]
def run(T,seed,nmeas,neq=1000,per=2):
    rnd=random.Random(seed);p=1-math.exp(-2/T)
    s=[1]*NN;M=NN;m2=0;m4=0;c=0
    for it in range(neq+nmeas*per):
        i=rnd.randrange(NN);sp=s[i];s[i]=-sp;st=[i];M-=2*sp
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rnd.random()<p:
                    s[k]=-sp;M-=2*sp;st.append(k)
        if it>=neq and (it-neq)%per==per-1:
            m=M/NN;m2+=m*m;m4+=m**4;c+=1
    return m2/c,m4/c
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={};R=10
for T in Ts:
    Us=[]
    for r in range(R):
        a,b=run(T,1000*r+int(T*10),30000)
        Us.append(1-b/(3*a*a))
    res[T]=(float(np.mean(Us)),float(np.std(Us,ddof=1)/math.sqrt(R)))
    print(T,res[T],flush=True)
json.dump(res,open('/work/res.json','w'))
