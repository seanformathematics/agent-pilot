
import numpy as np, random, time, os, sys
os.makedirs('/work/data', exist_ok=True)
N=16; L=N*N
nb=[]
for i in range(N):
    for j in range(N):
        nb.append((((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N))

def run_chain(T, nsteps, seed, block=1000, nburn=10000):
    rng = random.Random(seed)
    p = 1-np.exp(-2.0/T)
    rnd = rng.random; rr = rng.randrange
    s=[1 if rng.random()<0.5 else -1 for _ in range(L)]
    M = sum(s)
    def step(M):
        k0 = rr(L); sp=s[k0]; s[k0]=-sp; stack=[k0]; cnt=1
        while stack:
            k=stack.pop()
            for n in nb[k]:
                if s[n]==sp and rnd()<p:
                    s[n]=-sp; stack.append(n); cnt+=1
        return M - 2*sp*cnt
    for _ in range(nburn):
        M=step(M)
    nblk = nsteps//block
    out = np.zeros((nblk,3))
    for b in range(nblk):
        a2=a4=a1=0.0
        for _ in range(block):
            M=step(M)
            m=M/L
            m2=m*m
            a2+=m2; a4+=m2*m2; a1+=abs(m)
        out[b]=(a2,a4,a1)
    return out

if __name__=="__main__":
    Ts=[float(x) for x in sys.argv[1].split(',')]
    nsteps=int(sys.argv[2]); chains=[int(x) for x in sys.argv[3].split(',')]
    for T in Ts:
        t0=time.time()
        for c in chains:
            seed = 1000*int(round(T*100)) + c
            out = run_chain(T, nsteps, seed)
            np.save(f'/work/data/T{T:.2f}_c{c}.npy', out)
        print(T, 'done', time.time()-t0, flush=True)
