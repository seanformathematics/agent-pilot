import numpy as np, time, os
print(os.cpu_count(), len(os.sched_getaffinity(0)))
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
ev,od=make_masks()
def sweep(S,beta,rng,ev,od,ptab):
    for mask in (ev,od):
        nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
        dE=2*S*nn  # in {-8..8}
        acc=rng.random(S.shape)<ptab[dE+8]
        flip=acc&mask
        S[flip]*=-1
    return S
R=2000
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
for T in [2.269]:
    beta=1/T
    ptab=np.exp(-beta*np.clip(np.arange(-8,9),0,None)*1.0)
    ptab=np.array([min(1.0,np.exp(-beta*d)) for d in range(-8,9)])
    t0=time.time()
    for k in range(50):
        sweep(S,beta,rng,ev,od,ptab)
    print("time per sweep",(time.time()-t0)/50)