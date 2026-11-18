
import numpy as np, sys, time
sys.path.insert(0,'/work')
import ising
res = {}
for T in [1.5, 2.0, 2.3, 2.8]:
    t0=time.time()
    S,m,E = ising.run_chain(T, 256, 3000, seed=11, nburn=0, rec_E=True, N=16)
    res[T]=(m,E)
    m2 = m**2
    # time-dependence of replica-averaged m2 and E in windows
    w = [0,50,100,200,400,800,1600,2400]
    line=[]
    for a,b in zip(w[:-1],w[1:]):
        line.append("%d-%d: m2=%.4f E=%.1f"%(a,b,m2[a:b].mean(), E[a:b].mean()))
    print("T=%.2f time %.1fs"%(T,time.time()-t0))
    print("  "+"\n  ".join(line))
    # integrated autocorr time for m2 (avg over replicas), using second half
    x = m2[1000:]
    x = x - x.mean(axis=0)
    n = x.shape[0]
    def tau_int(x):
        # FFT-based autocorrelation averaged over replicas
        f = np.fft.rfft(x, n=2*n, axis=0)
        ac = np.fft.irfft(f*np.conj(f), axis=0)[:n]
        ac = ac.mean(axis=1)
        ac = ac/ac[0]
        # sum until first negative
        t = 1.0
        for k in range(1,n):
            if ac[k] < 0: break
            t += 2*ac[k]
        return t
    print("  tau_int(m^2) ~ %.1f sweeps"%tau_int(x))
