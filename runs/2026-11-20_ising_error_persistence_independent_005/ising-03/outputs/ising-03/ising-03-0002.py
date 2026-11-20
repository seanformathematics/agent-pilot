
import random, math, time, os
os.makedirs('/work',exist_ok=True)
src = r'''
import random, math, sys, time, json
N=16; NN=N*N
nb=[]
for i in range(N):
    for j in range(N):
        s=i*N+j
        nb.append((((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N))
def run(T, seed, nsweeps, nburn, out_path=None):
    rng=random.Random(seed)
    rnd=rng.random
    padd=1.0-math.exp(-2.0/T)
    spin=[1]*NN
    M=NN
    def wolff():
        nonlocal M
        s0=int(rnd()*NN)
        c=spin[s0]
        spin[s0]=-c
        stack=[s0]
        cl=[s0]
        while stack:
            s=stack.pop()
            for t in nb[s]:
                if spin[t]==c and rnd()<padd:
                    spin[t]=-c
                    stack.append(t); cl.append(t)
        M += -2*c*len(cl)
    return wolff, lambda: M, spin
'''
open('/work/ising_lib.py','w').write(src)
exec(src)
wolff,getM,spin=run(2.269,1,0,0)
t=time.time()
for _ in range(20000):
    wolff()
dt=time.time()-t
print('per cluster us', dt/20000*1e6, 'M',getM(), sum(spin))
