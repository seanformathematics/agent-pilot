
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def ld(fn): return pickle.load(open(fn,'rb'))
for T in (2.8,2.5):
    k=Ts.index(T)
    old=ld(f'/work/metro_k{k}.pkl'); new=ld(f'/work/metro2_k{k}.pkl')
    print(f"===== T={T}")
    for name,d in (("old (cold, 2000 burn-in)",old),("new (hot, 3000 burn-in)",new)):
        b2=d['blk'][0]; b4=d['blk'][1]; nb=b2.shape[1]; R=b2.shape[0]
        # U per time-segment of 40 blocks (4000 sweeps): ensemble means
        segs=[]
        for s in range(0,nb,40):
            U,e=sw.U_jack(b2[:,s:s+40].mean(axis=1), b4[:,s:s+40].mean(axis=1)); segs.append((U,e))
        print(name, " U per 4000-sweep segment:", " ".join(f"{u:.4f}" for u,_ in segs), f"  (seg err ~{np.mean([e for _,e in segs]):.4f})")
        print("    segment-to-segment std = %.5f (expected ~ seg err %.5f)"%(np.std([u for u,_ in segs],ddof=1), np.mean([e for _,e in segs])))
        # group scatter: 20 groups of 50 replicas
        s2=b2.mean(axis=1); s4=b4.mean(axis=1)
        rng=np.random.default_rng(1); perm=rng.permutation(R); s2p=s2[perm]; s4p=s4[perm]
        Ug=[];eg=[]
        for g in range(20):
            sl=slice(g*50,(g+1)*50); U,e=sw.U_jack(s2p[sl],s4p[sl]); Ug.append(U); eg.append(e)
        print("    group scatter std=%.5f vs rms jackknife err of group=%.5f  ratio=%.2f"%(np.std(Ug,ddof=1),np.sqrt(np.mean(np.square(eg))),np.std(Ug,ddof=1)/np.sqrt(np.mean(np.square(eg)))))
        U,e=sw.U_jack(s2,s4)
        print(f"    overall U={U:.5f}+-{e:.5f}; <m2>={s2.mean():.5f}+-{s2.std(ddof=1)/np.sqrt(R):.5f}; <m4>={s4.mean():.6f}+-{s4.std(ddof=1)/np.sqrt(R):.6f}")
    # time-series of ensemble m2 in old run: first vs last blocks
    o2=old['blk'][0].mean(axis=0); n2=new['blk'][0].mean(axis=0)
    print("    ensemble-mean m2 per 20-block chunk: old", np.array([o2[i:i+20].mean() for i in range(0,200,20)]).round(5))
    print("                                          new", np.array([n2[i:i+20].mean() for i in range(0,200,20)]).round(5))
