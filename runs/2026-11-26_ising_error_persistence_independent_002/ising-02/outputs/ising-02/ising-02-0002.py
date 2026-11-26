
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    b=((i+j)%2==0)
    return b, ~b
def run(T, R, seed, n_eq, n_meas, out=None):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    black,white=make_masks()
    # acceptance table indexed by dE in {-8,-4,0,4,8} -> use s*nnsum
    # dE = 2*s*h, h = nn sum in {-4..4}
    acc=np.zeros(9)
    for h in range(-4,5):
        for sv in (-1,1):
            dE=2*sv*h
            acc[(h+4)]= 1.0 if dE<=0 else np.exp(-dE/T)
    # per-site acceptance prob for given spin and h: precompute as function of (s*h)
    p_tab=np.zeros(9)
    for k in range(-4,5):
        dE=2*k
        p_tab[k+4]=1.0 if dE<=0 else np.exp(-dE/T)
    def sweep(s):
        for mask in (black,white):
            h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            k = (s*h).astype(np.int16)  # s*h in -4..4
            p = p_tab[k+4]
            u = rng.random(s.shape)
            flip = (u < p) & mask
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt, s

if __name__=="__main__":
    t=time.time()
    m2,m4,s=run(2.3,1000,1,50,50)
    dt=time.time()-t
    print("time per sweep (R=1000):", dt/100)
    print(m2.mean(), m4.mean())
