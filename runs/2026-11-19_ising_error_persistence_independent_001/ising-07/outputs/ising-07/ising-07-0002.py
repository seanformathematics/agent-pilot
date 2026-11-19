import random, math, time, os
os.makedirs('/work', exist_ok=True)
N=16; NN=N*N
nbr=[]
for i in range(N):
    for j in range(N):
        nbr.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])

def run(T, nsweeps, seed, neq):
    rng=random.Random(seed)
    rnd=rng.random
    s=[1 if rnd()<0.5 else -1 for _ in range(NN)]
    padd=1-math.exp(-2.0/T)
    M=sum(s)
    out_m=[]
    for step in range(nsweeps+neq):
        i0=int(rnd()*NN)
        sd=s[i0]
        cl=[i0]; s[i0]=-sd; stack=[i0]
        while stack:
            k=stack.pop()
            for nb in nbr[k]:
                if s[nb]==sd and rnd()<padd:
                    s[nb]=-sd; stack.append(nb); cl.append(nb)
        M-=2*sd*len(cl)
        if step>=neq:
            out_m.append(M/NN)
    return out_m

t=time.time()
m=run(2.269,20000,12345,2000)
print("time",time.time()-t, len(m))
import statistics
m2=[x*x for x in m]; m4=[x**4 for x in m]
U=1-statistics.fmean(m4)/(3*statistics.fmean(m2)**2)
print("U",U, "mean|m|",statistics.fmean([abs(x) for x in m]))
