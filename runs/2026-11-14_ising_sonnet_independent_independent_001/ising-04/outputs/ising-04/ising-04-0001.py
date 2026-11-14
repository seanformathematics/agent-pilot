
import random, math, time, numpy as np, json, sys
N=16;NN=N*N
nb=[[((i//N+1)%N)*N+i%N,((i//N-1)%N)*N+i%N,(i//N)*N+(i%N+1)%N,(i//N)*N+(i%N-1)%N] for i in range(NN)]
def run(T,seed,nmeas,neq=500,per=2):
    rnd=random.Random(seed);p=1-math.exp(-2/T)
    s=[1]*NN if T<2.3 else [rnd.choice((-1,1)) for _ in range(NN)]
    m2=[];m4=[]
    M=sum(s)
    for it in range(neq+nmeas*per):
        i=rnd.randrange(NN);sp=s[i];s[i]=-sp;st=[i];M-=2*sp
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rnd.random()<p:
                    s[k]=-sp;M-=2*sp;st.append(k)
        if it>=neq and (it-neq)%per==per-1:
            m=M/NN;m2.append(m*m);m4.append(m**4)
    return np.array(m2),np.array(m4)
t=time.time();run(2.3,1,2000);print(time.time()-t)
