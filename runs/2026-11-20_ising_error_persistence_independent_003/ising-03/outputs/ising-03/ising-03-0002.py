import time, numpy as np
R,N=2000,16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
mask=np.zeros((N,N),bool); mask[(np.arange(N)[:,None]+np.arange(N)[None,:])%2==0]=True
def sweep(s,T,rng):
    for m in (mask,~mask):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        p=np.exp(-dE/T)
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<p)
        s[acc&m]*=-1
    return s
t=time.time()
for _ in range(20): sweep(s,2.27,rng)
print("time per sweep",(time.time()-t)/20)
