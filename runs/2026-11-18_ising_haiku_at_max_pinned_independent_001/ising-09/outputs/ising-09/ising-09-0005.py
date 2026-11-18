
import os, glob, numpy as np
L=16; N2=L*L
TEMPS=[round(1.5+0.1*k,1) for k in range(14)]
def tau_int(x, maxlag=5000):
    x = np.asarray(x, dtype=np.float64); x = x - x.mean(); n=len(x)
    f = np.fft.rfft(x, 2*n); acf = np.fft.irfft(f*np.conj(f))[:n]; acf/=acf[0]
    tau=0.5
    for w in range(1, min(maxlag, n//2)):
        tau += acf[w]
        if w >= 6*tau: return tau
    return tau

def jk_U(chains, nb=100):
    # blocked jackknife across all chains: each chain split into nb/len blocks
    blocks2=[]; blocks4=[]
    for M in chains:
        m = M.astype(np.float64)/N2; m2=m*m; m4=m2*m2
        k = max(1, nb//len(chains))
        bs = len(m)//k
        blocks2.append(m2[:bs*k].reshape(k,bs).sum(1)); blocks4.append(m4[:bs*k].reshape(k,bs).sum(1))
        blocks_n = None
    s2=np.concatenate(blocks2); s4=np.concatenate(blocks4)
    nbtot = len(s2)
    nsamp = np.array([0])
    # sample counts per block (equal size within chain, chains may differ in length -> they are equal here)
    S2=s2.sum(); S4=s4.sum(); n_tot_per_block = None
    # all blocks have equal length in this design
    nper = sum(len(M) for M in chains)/nbtot
    NN = nper*nbtot
    U = 1 - (S4/NN)/(3*(S2/NN)**2)
    Uj=np.empty(nbtot)
    for b in range(nbtot):
        a2=(S2-s2[b])/(NN-nper); a4=(S4-s4[b])/(NN-nper)
        Uj[b]=1-a4/(3*a2*a2)
    err = np.sqrt((nbtot-1)/nbtot*np.sum((Uj-Uj.mean())**2))
    return U, err

print(f"{'T':>5} {'nch':>3} {'U_pooled':>9} {'jk_err':>8} {'chainU_std':>10} {'tau(m2)':>8} {'tau(m4)':>8} {'chainU':>s}")
for T in TEMPS:
    files = sorted(glob.glob(f"/work/data/M_T{T:.1f}_c*.npy"))
    chains = [np.load(f).astype(np.int64) for f in files]
    if not chains: continue
    U, err = jk_U(chains, nb=100)
    cu=[]
    taus2=[]; taus4=[]
    for M in chains:
        m=M/N2; m2=m*m; m4=m2*m2
        cu.append(1-m4.mean()/(3*m2.mean()**2))
        taus2.append(tau_int(m2)); taus4.append(tau_int(m4))
    cu=np.array(cu)
    sd = cu.std(ddof=1)/np.sqrt(len(cu)) if len(cu)>1 else float('nan')
    print(f"{T:5.1f} {len(chains):3d} {U:9.5f} {err:8.5f} {sd:10.5f} {np.mean(taus2):8.1f} {np.mean(taus4):8.1f}  " + " ".join(f"{x:.4f}" for x in cu))
