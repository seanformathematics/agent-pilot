
import glob, numpy as np, json
L=16; N2=L*L
TEMPS=[round(1.5+0.1*k,1) for k in range(14)]
def tau_int(x, maxlag=5000):
    x=np.asarray(x,dtype=np.float64); x=x-x.mean(); n=len(x)
    f=np.fft.rfft(x,2*n); acf=np.fft.irfft(f*np.conj(f))[:n]; acf/=acf[0]
    tau=0.5
    for w in range(1,min(maxlag,n//2)):
        tau+=acf[w]
        if w>=6*tau: return tau
    return tau
results={}
print(f"{'T':>4} {'K':>2} {'nsamp':>9} {'U':>9} {'jk_err':>8} {'bias_jk':>9} {'chain_sd/rtK':>12} {'drift(1st-2nd half)':>20} {'tau_m2':>6}")
for T in TEMPS:
    files=sorted(glob.glob(f"/work/data/M_T{T:.1f}_c*.npy"))
    chains=[np.load(f).astype(np.int64) for f in files]
    K=len(chains)
    B=50  # blocks per chain (10^4 steps each)
    s2=[]; s4=[]; nb_len=[]
    chainU=[]; drift=[]
    for M in chains:
        m=M/N2; m2=m*m; m4=m2*m2
        bs=len(m)//B
        s2.append(m2[:bs*B].reshape(B,bs).sum(1)); s4.append(m4[:bs*B].reshape(B,bs).sum(1))
        nb_len.append(np.full(B,bs))
        chainU.append(1-m4.mean()/(3*m2.mean()**2))
        h=len(m)//2
        U1=1-(m4[:h].mean())/(3*m2[:h].mean()**2); U2=1-(m4[h:].mean())/(3*m2[h:].mean()**2)
        drift.append(U1-U2)
    s2=np.concatenate(s2); s4=np.concatenate(s4); bl=np.concatenate(nb_len).astype(float)
    S2=s2.sum(); S4=s4.sum(); NN=bl.sum()
    U=1-(S4/NN)/(3*(S2/NN)**2)
    nbk=len(s2)
    Uj=np.empty(nbk)
    for b in range(nbk):
        a2=(S2-s2[b])/(NN-bl[b]); a4=(S4-s4[b])/(NN-bl[b])
        Uj[b]=1-a4/(3*a2*a2)
    err=np.sqrt((nbk-1)/nbk*np.sum((Uj-Uj.mean())**2))
    bias=(nbk-1)*(Uj.mean()-U)
    chainU=np.array(chainU)
    csd=chainU.std(ddof=1)/np.sqrt(K) if K>1 else float('nan')
    tau2=np.mean([tau_int((M/N2)**2) for M in chains])
    results[T]=dict(U=float(U), err=float(err), n=int(NN), K=K, bias=float(bias), chain_err=float(csd),
                    seeds=[100000+int(round(T*10))*100+int(f.split('_c')[1].split('.npy')[0]) for f in files])
    print(f"{T:4.1f} {K:2d} {int(NN):9d} {U:9.5f} {err:8.5f} {bias:9.2e} {csd:12.5f} {np.mean(drift):20.5f} {tau2:6.2f}")
json.dump(results, open('/work/results.json','w'), indent=1)
print("U within (-inf, 2/3]:", all(np.isfinite(r['U']) and r['U']<=2/3 for r in results.values()))
print("max err:", max(r['err'] for r in results.values()))
