code = r'''
import math, random, sys, time, numpy as np
N=16; L=N*N
nb=[]
for i in range(L):
    r,c=divmod(i,N)
    nb.append((((r+1)%N)*N+c,((r-1)%N)*N+c,r*N+(c+1)%N,r*N+(c-1)%N))
def run(T, seed, time_budget, nburn=2000, maxrec=None):
    rng=random.Random(seed)
    s=[rng.choice((-1,1)) for _ in range(L)]
    M=sum(s)
    p=1-math.exp(-2.0/T)
    rnd=rng.random; rr=rng.randrange
    rec=[]
    t0=time.time()
    k=0
    while True:
        i=rr(L); sg=s[i]; s[i]=-sg; stack=[i]; size=1
        while stack:
            q=stack.pop()
            for j in nb[q]:
                if s[j]==sg and rnd()<p:
                    s[j]=-sg; stack.append(j); size+=1
        M-=2*sg*size
        k+=1
        if k>nburn: rec.append(M)
        if (k & 1023)==0 and time.time()-t0>time_budget: break
    return np.array(rec,dtype=np.int32), k
'''
open('/work/wolff.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from wolff import run
import numpy as np, time
def stats(M):
    m=M/256.0
    m2=(m*m).mean(); m4=(m**4).mean()
    return 1-m4/(3*m2*m2)
for T in [1.9,2.269]:
    t=time.time()
    M,k=run(T,12345,20)
    print(T,'clusters',k,'rec',len(M),'time',time.time()-t,'U',stats(M.astype(float)))
    # autocorrelation of M^2 via FFT
    x=(M/256.0)**2; x=x-x.mean()
    n=len(x); f=np.fft.rfft(x,2*n); ac=np.fft.irfft(f*np.conj(f))[:n]; ac/=ac[0]
    tau=1+2*np.sum(ac[1:next(i for i in range(1,n) if ac[i]<0.0 or i>5000)])
    print('tau_int(m^2) in clusters ~',tau, 'mean |m|', np.abs(M/256).mean())
'''
'''