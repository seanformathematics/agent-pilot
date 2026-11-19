
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
def jk_mean(x):
    return x.mean(), x.std(ddof=1)/np.sqrt(len(x))
for k in range(8,14):
    T=Ts[k]
    A=load('A',k); B=load('B',k)
    blk=np.concatenate([A['blk'],B['blk']],axis=1)
    M=pickle.load(open(f'/work/metro_k{k}.pkl','rb'))
    s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1); i2=blk[2].mean(axis=1); i4=blk[3].mean(axis=1)
    m2s,e2s=jk_mean(s2); m4s,e4s=jk_mean(s4); m2i,e2i=jk_mean(i2); m4i,e4i=jk_mean(i4)
    ms2=M['blk'][0].mean(axis=1); ms4=M['blk'][1].mean(axis=1)
    m2m,e2m=jk_mean(ms2); m4m,e4m=jk_mean(ms4)
    print(f"T={T}: <m2> SW={m2s:.5f}+-{e2s:.5f} SWimpr={m2i:.5f}+-{e2i:.5f} Metro={m2m:.5f}+-{e2m:.5f} | z(M-SW)={(m2m-m2s)/np.hypot(e2m,e2s):+.2f}, z(M-SWimpr)={(m2m-m2i)/np.hypot(e2m,e2i):+.2f}")
    print(f"        <m4> SW={m4s:.6f}+-{e4s:.6f} SWimpr={m4i:.6f}+-{e4i:.6f} Metro={m4m:.6f}+-{e4m:.6f} | z(M-SW)={(m4m-m4s)/np.hypot(e4m,e4s):+.2f}, z(M-SWimpr)={(m4m-m4i)/np.hypot(e4m,e4i):+.2f}")
    Um,em=sw.U_jack(ms2,ms4); Ui,ei=sw.U_jack(i2,i4); Up,ep=sw.U_jack(s2,s4)
    print(f"        U: SW plain {Up:.5f}+-{ep:.5f}, SW impr {Ui:.5f}+-{ei:.5f}, Metro {Um:.5f}+-{em:.5f}; z(M-impr)={(Um-Ui)/np.hypot(em,ei):+.2f}")
    # per-replica outliers in Metropolis
    z=(ms2-ms2.mean())/ms2.std(); z4=(ms4-ms4.mean())/ms4.std()
    print(f"        Metro per-replica m2 max|z|={np.abs(z).max():.2f}, m4 max|z|={np.abs(z4).max():.2f}; SW per-replica m2 max|z|={np.abs((s2-s2.mean())/s2.std()).max():.2f}")
