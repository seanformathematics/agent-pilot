
import numpy as np, glob
files={1.9:'/work/res_T1.9_s201.npz',2.0:'/work/res_T2.0_s202.npz',2.1:'/work/res_T2.1_s203.npz',
       2.2:'/work/res_T2.2_s204.npz',2.3:'/work/res_T2.3_s101.npz',2.4:'/work/res_T2.4_s205.npz'}
out=[]
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(files[T]); n=int(d['nmeas'])
    S2=d['S2']/n; S4=d['S4']/n; R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    G=50; idx=np.arange(R)%G  # group jackknife
    Us=[]
    for g in range(G):
        k=idx!=g
        Us.append(1-S4[k].mean()/(3*S2[k].mean()**2))
    Us=np.array(Us)
    err=np.sqrt((G-1)/G*np.sum((Us-Us.mean())**2))
    blk=d['blk']
    print(f"T={T} U={U:.5f} err={err:.5f}  block std={blk.std(ddof=1):.5f} blk mean={blk.mean():.5f} n_meas={n} R={R} seed={int(d['seed'])}")
    out.append((T,U,err,int(d['seed'])))
np.save('/work/summary.npy',np.array(out))
