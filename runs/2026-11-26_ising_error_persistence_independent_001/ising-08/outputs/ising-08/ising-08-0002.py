import numpy as np, time, math, random
N=16; L=N*N
nb=[[((i//N+di)%N)*N+(i%N+dj)%N for di,dj in ((1,0),(-1,0),(0,1),(0,-1))] for i in range(L) for _ in [0]]
nb=[]
for i in range(L):
    r,c=divmod(i,N)
    nb.append([((r+1)%N)*N+c,((r-1)%N)*N+c,r*N+(c+1)%N,r*N+(c-1)%N])
def wolff(s,T,rng,nsteps):
    p=1-math.exp(-2/T)
    ms=[]
    for _ in range(nsteps):
        i=rng.randrange(L); sg=s[i]; s[i]=-sg; stack=[i]; 
        while stack:
            k=stack.pop()
            for j in nb[k]:
                if s[j]==sg and rng.random()<p:
                    s[j]=-sg; stack.append(j)
    return s
rng=random.Random(1)
s=[rng.choice((-1,1)) for _ in range(L)]
t=time.time(); wolff(s,2.269,rng,2000); print('wolff clusters/s',2000/(time.time()-t))
# vectorized metropolis
R=1000
rs=np.random.default_rng(2)
S=rs.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T):
    for m in masks:
        nbsum=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nbsum
        acc=(dE<=0)|(rs.random(S.shape)<np.exp(-dE/T))
        flip=acc&m
        S[flip]*=-1
t=time.time()
for _ in range(20): sweep(S,2.269)
print('metro replica-sweeps/s',20*R/(time.time()-t))