import numpy as np
for T in [1.9,2.0,2.1]:
    d=np.load(f'/work/N16_T{T}.npz'); S2=d['S2'];S4=d['S4']
    Ur=1-S4/(3*S2**2)
    R=len(S2)
    print(T,'per-replica U mean',Ur.mean(),'sd',Ur.std(ddof=1),'sem',Ur.std(ddof=1)/np.sqrt(R))
    print('  m2 mean',S2.mean(),'sd per replica',S2.std(ddof=1))
    # bootstrap over replicas
    rng=np.random.default_rng(0)
    bs=[]
    for b in range(300):
        idx=rng.integers(0,R,R)
        bs.append(1-S4[idx].mean()/(3*S2[idx].mean()**2))
    print('  bootstrap U err',np.std(bs))