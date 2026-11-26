
import numpy as np, math
N=16
def ref(T, nsw, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice([-1,1],size=(N,N))
    out=[]
    for sw in range(nsw):
        for i in range(N):
            for j in range(N):
                h=s[(i+1)%N,j]+s[(i-1)%N,j]+s[i,(j+1)%N]+s[i,(j-1)%N]
                dE=2*s[i,j]*h
                if dE<=0 or rng.random()<math.exp(-dE/T):
                    s[i,j]=-s[i,j]
        E=-(np.sum(s*np.roll(s,1,0))+np.sum(s*np.roll(s,1,1)))/(N*N)
        out.append((round(E,3), round(abs(s.mean()),3)))
    return out[-5:]
for T in [1.0, 5.0, 2.4]:
    print(T, ref(T, 200, 2))
