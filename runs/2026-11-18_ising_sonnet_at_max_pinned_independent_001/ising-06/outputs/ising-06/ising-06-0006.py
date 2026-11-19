
import sys, time
sys.path.insert(0,'/work')
import numpy as np, sw, exact_torus as ex
N=16

def per_step_series(T, R, nsteps, seed, start):
    b = sw.SWBatch(N, R, T, seed, start)
    E = np.zeros((nsteps,R)); M2 = np.zeros((nsteps,R)); I2 = np.zeros((nsteps,R)); I4=np.zeros((nsteps,R)); M4=np.zeros((nsteps,R))
    for t in range(nsteps):
        S2,S4 = b.step()
        m,e = b.observables()
        E[t]=e; M2[t]=m*m; M4[t]=m**4; I2[t]=S2/N**4; I4[t]=(3*S2**2-2*S4)/N**8
    return E,M2,M4,I2,I4

t0=time.time()
print("=== relaxation from hot/cold start (R=2048): mean energy and mean improved m2 vs step ===")
for T in [1.5, 2.0, 2.27, 2.6]:
    ee = ex.energy_per_spin_exact(T,N)
    Eh,_,_,Ih,_ = per_step_series(T, 2048, 41, 11, "hot")
    Ec,_,_,Ic,_ = per_step_series(T, 2048, 41, 12, "cold")
    print(f"T={T}  exact e={ee:.5f}   (statistical error of each mean ~ {Eh[20:].std(axis=1).mean()/np.sqrt(2048):.4f})")
    print("  step : e_hot   e_cold  | m2_hot  m2_cold")
    for t in [0,1,2,3,4,5,6,8,10,15,20,30,40]:
        print(f"  {t:3d}  : {Eh[t].mean():.4f} {Ec[t].mean():.4f} | {Ih[t].mean():.4f} {Ic[t].mean():.4f}")
    # late-time averages
    print("  mean over steps 20..40: e_hot %.5f e_cold %.5f ; m2_hot %.5f m2_cold %.5f" % (Eh[20:].mean(),Ec[20:].mean(),Ih[20:].mean(),Ic[20:].mean()))
print("elapsed", time.time()-t0, flush=True)

def tau_int(x, cmax=6.0):
    """x: (n, R) series, R independent replicas. Autocorrelation averaged over replicas; Sokal window."""
    n,R = x.shape
    mu = x.mean()
    y = x - mu
    f = np.fft.rfft(y, n=2*n, axis=0)
    ac = np.fft.irfft(f*np.conj(f), axis=0)[:n].sum(axis=1)   # sum over replicas
    ac = ac / (np.arange(n,0,-1)[:,None].ravel()*R)
    rho = ac/ac[0]
    tau = 0.5
    for W in range(1, n//2):
        tau = 0.5 + rho[1:W+1].sum()
        if W >= cmax*tau: break
    return tau, W

print("=== integrated autocorrelation times (in SW updates), R=128, 3000 steps ===")
for T in [1.5, 2.0, 2.27, 2.5, 2.8]:
    E,M2,M4,I2,I4 = per_step_series(T, 128, 3000, 77, "hot")
    out=[]
    for name,x in [("e",E),("m2",M2),("m4",M4),("i2",I2),("i4",I4)]:
        tau,W = tau_int(x[200:])
        out.append(f"{name}: {tau:.2f}")
    print(f"T={T}: "+"  ".join(out), flush=True)
print("elapsed", time.time()-t0)
