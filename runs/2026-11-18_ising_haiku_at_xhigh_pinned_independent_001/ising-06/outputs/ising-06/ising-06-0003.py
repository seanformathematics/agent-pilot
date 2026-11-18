import numpy as np, time
N=16
def make_table(T):
    tab = np.zeros(9, dtype=np.float32)
    for p in range(-4,5):
        tab[p+4] = 1.0 if p<=0 else np.exp(-2.0*p/T)
    return tab
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==c) for c in (0,1)]
def run(R, T, sweeps, seed, burn=0):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R,N,N))<0.5, -1, 1).astype(np.int8)
    tab = make_table(T)
    M2 = np.zeros((sweeps,R)); M4=np.zeros((sweeps,R))
    for sw in range(sweeps):
        for c in (0,1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            pt = tab[(s*nb + 4)]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pt) & masks[c]
            s = np.where(flip, -s, s).astype(np.int8)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        M2[sw]=m*m; M4[sw]=m**4
    return M2, M4
def tauint(x, maxlag=2000):
    x = x - x.mean()
    n=len(x)
    f = np.fft.rfft(x, 2*n)
    ac = np.fft.irfft(f*np.conj(f))[:n]
    ac/=ac[0]
    # automatic windowing
    t=0.5
    for k in range(1,min(maxlag,n//4)):
        t += ac[k]
        if k >= 6*t: return t, k
    return t, -1
for T in [1.5, 2.27, 2.8]:
    t0=time.time()
    M2,M4 = run(400, T, 3000, 12345)
    dt=time.time()-t0
    blk = M2.reshape(15,200,400).mean(axis=(1,2))
    print("T",T,"time",round(dt,1))
    print(" <m^2> per 200-sweep block:", np.round(blk,4))
    U_blocks = []
    for b in range(15):
        a2=M2[b*200:(b+1)*200].mean(); a4=M4[b*200:(b+1)*200].mean()
        U_blocks.append(1-a4/(3*a2*a2))
    print(" U per block:", np.round(U_blocks,4))
    # tau for pooled series over replicas, using second half
    x2 = M2[1000:].mean(axis=1); x4 = M4[1000:].mean(axis=1)
    print(" tau m2:", tauint(x2), " tau m4:", tauint(x4))
    print(" tau |m| series (replica0):", tauint(np.sqrt(M2[1000:,0])))
