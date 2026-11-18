
import random, math, time
import numpy as np
exec(open('/dev/stdin').read()) if False else None
N=16
NS=N*N
nb=[]
for x in range(N):
    for y in range(N):
        nb.append([((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N])

def run_wolff(T, nsteps, nburn, seed, start=1):
    rng=random.Random(seed)
    rnd=rng.random; rint=rng.randrange
    s=[start]*NS if start in (1,-1) else [1 if rnd()<0.5 else -1 for _ in range(NS)]
    padd=1.0-math.exp(-2.0/T)
    M=sum(s)
    out=np.empty(nsteps, dtype=np.int32)
    total=nburn+nsteps
    for step in range(total):
        i=rint(NS)
        sg=s[i]
        s[i]=-sg
        stack=[i]
        size=1
        while stack:
            k=stack.pop()
            for j in nb[k]:
                if s[j]==sg and rnd()<padd:
                    s[j]=-sg
                    stack.append(j)
                    size+=1
        M-=2*sg*size
        if step>=nburn:
            out[step-nburn]=M
    return out

def U_of(M):
    m=M/NS
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2**2)

def tau_int(x, maxlag=2000):
    x=x-x.mean()
    n=len(x)
    f=np.fft.rfft(x, 2*n)
    ac=np.fft.irfft(f*np.conj(f))[:n]
    ac/=ac[0]
    # Sokal automatic window
    tau=0.5
    for w in range(1,maxlag):
        tau+=ac[w]
        if w>=6*tau:
            return tau, w
    return tau, maxlag

t=time.time()
out=run_wolff(2.269, 1000000, 5000, 12345, start=1)
print("time", time.time()-t)
m=out/NS
for name,x in [("m2",m**2),("m4",m**4),("|m|",np.abs(m))]:
    print(name, tau_int(x))
U=U_of(out)
print("U full", U)
# binned errors
for nb_ in [10,100,1000,10000]:
    L=len(out)//nb_
    Ub=[U_of(out[k*L:(k+1)*L]) for k in range(nb_)]
    print("nbins",nb_,"binlen",L,"err", np.std(Ub,ddof=1)/np.sqrt(nb_))
np.save('/work/pilot_2269.npy', out) if False else None
