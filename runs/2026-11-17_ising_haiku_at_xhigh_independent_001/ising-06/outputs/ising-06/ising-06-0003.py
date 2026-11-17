import numpy as np, time
N=16
def make_tables(T):
    xs = np.array([-4,-2,0,2,4])
    p = np.where(xs<=0, 1.0, np.exp(-2.0*xs/T))
    return p.astype(np.float32)
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def run(R, T, sweeps, seed):
    rng = np.random.default_rng(seed)
    S = np.ones((R,N,N), dtype=np.int8)
    S[R//2:] = -1
    tab = make_tables(T)
    out = np.zeros((sweeps,R))
    for t in range(sweeps):
        for mk in masks:
            h = np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            x = S*h
            pr = tab[(x+4)>>1]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pr) & mk
            np.negative(S, out=S, where=flip)
        out[t]=S.reshape(R,-1).mean(axis=1)
    return out
def tau_int(x, maxlag=500):
    # x: (T,) series; normalized autocorrelation, windowed
    x = x - x.mean()
    n=len(x)
    f = np.fft.rfft(x, 2*n)
    ac = np.fft.irfft(f*np.conj(f))[:n]
    ac/= ac[0]
    tau=0.5
    for k in range(1,min(maxlag,n//4)):
        tau += ac[k]
        if k > 6*tau: break
    return tau
for T in [2.27, 2.0]:
    t0=time.time()
    m = run(200, T, 4000, 12345)
    print("T",T,"time",time.time()-t0)
    m2=m**2; m4=m**4
    # running block means of U over time for equilibration check
    for blk in [0,250,500,1000,2000,3000]:
        seg=slice(blk, blk+500) if blk+500<=4000 else slice(3500,4000)
        U = 1-m4[seg].mean(0)/(3*m2[seg].mean(0)**2)
        print(" block start",blk," <U> over replicas:", U.mean().round(4), " <|m|>", np.abs(m[seg]).mean().round(4))
    # tau for m2 and m4 averaged over replicas, after 1000 burn-in
    taus2=[tau_int(m2[1000:,r]) for r in range(0,200,10)]
    taus4=[tau_int(m4[1000:,r]) for r in range(0,200,10)]
    print(" tau_int m2 mean", np.mean(taus2).round(1), " m4", np.mean(taus4).round(1))
    # sign flips count
    sgn = np.sign(m[1000:])
    flips = (np.diff(sgn,axis=0)!=0).sum()
    print(" sign flips (post-burn) total:", flips)
