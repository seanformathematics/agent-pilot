
import numpy as np
for T in [1.9,2.0,2.1,2.2]:
    d=np.load(f'/work/res_T{T}.npz')
    a2,a4,b=d['a2'],d['a4'],d['blocks']
    Ui=1-a4/(3*a2**2)
    U=1-a4.mean()/(3*a2.mean()**2)
    print(T,"U=%.5f  std(Ui)/sqrt(R)=%.6f  std(a2)/sqrt(R)/mean=%.5f std(a4)/sqrt(R)/mean=%.5f"%(U, Ui.std()/np.sqrt(len(Ui)), a2.std()/np.sqrt(len(a2))/a2.mean(), a4.std()/np.sqrt(len(a4))/a4.mean()))
    # block-based: use the 20 blocks per replica of m2 means, estimate spread of block means
    bm=b.mean(axis=(0))  # mean over replicas per block
    print("   block-to-block std of replica-avg m2 (x sqrt(nblocks)):", bm.std()*np.sqrt(len(bm)) , "mean", bm.mean())
