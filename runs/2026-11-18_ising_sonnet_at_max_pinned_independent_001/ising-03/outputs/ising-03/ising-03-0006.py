
import sys, time
sys.path.insert(0,'/work')
import numpy as np, importlib
import sw, exact
importlib.reload(sw)

# ---- exact N=4 ----
N=4; n=16
states = np.arange(2**n, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(n,dtype=np.uint32)[None,:]) & 1).astype(np.int8)
s = (2*bits-1).reshape(-1,N,N).astype(np.int16)
E = -(s*np.roll(s,-1,axis=2) + s*np.roll(s,-1,axis=1)).sum(axis=(1,2)).astype(np.float64)
mm = s.sum(axis=(1,2))/float(n)
def ex(T):
    w=np.exp(-(E-E.min())/T); Z=w.sum()
    m2=(w*mm**2).sum()/Z; m4=(w*mm**4).sum()/Z; e=(w*E).sum()/Z/n
    return m2,m4,1-m4/(3*m2**2),e

t0=time.time()
zrec={'e':[], 'm2':[], 'm4':[], 'U':[], 'Ui':[]}
seedbase=777000
for T in [1.5,2.0,2.27,3.0,4.0]:
    m2e,m4e,Ue,ee=ex(T)
    for rep in range(10):
        R=2000; M=2000
        S=sw.IsingSW(4,R,T,seed=seedbase); seedbase+=1
        for _ in range(50): S.sweep()
        s2=np.zeros(R); s4=np.zeros(R); i2=np.zeros(R); i4=np.zeros(R); es=np.zeros(R)
        for _ in range(M):
            a,b=S.sweep(improved=True)
            m=S.magnetization()
            s2+=m**2; s4+=m**4; i2+=a; i4+=b; es+=S.energy_per_site()
        s2/=M; s4/=M; i2/=M; i4/=M; es/=M
        Up,ep=sw.U_jack(s2,s4); Ui,ei=sw.U_jack(i2,i4)
        zrec['U'].append((Up-Ue)/ep); zrec['Ui'].append((Ui-Ue)/ei)
        zrec['e'].append((es.mean()-ee)/(es.std(ddof=1)/np.sqrt(R)))
        zrec['m2'].append((s2.mean()-m2e)/(s2.std(ddof=1)/np.sqrt(R)))
        zrec['m4'].append((s4.mean()-m4e)/(s4.std(ddof=1)/np.sqrt(R)))
    print(f"T={T}: done  ({time.time()-t0:.0f}s)")
for k,v in zrec.items():
    v=np.array(v)
    print(f"{k:3s}: n={len(v)} mean z={v.mean():+.3f} (se {1/np.sqrt(len(v)):.3f}), rms z={np.sqrt((v**2).mean()):.3f}, max|z|={np.abs(v).max():.2f}")
# per-T mean z for energy
ze=np.array(zrec['e']).reshape(5,10)
print("per-T mean z for e:", ze.mean(axis=1).round(2), "(se 0.32 each)")
zu=np.array(zrec['U']).reshape(5,10)
print("per-T mean z for U:", zu.mean(axis=1).round(2))
