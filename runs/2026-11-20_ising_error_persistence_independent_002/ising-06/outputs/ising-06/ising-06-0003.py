
import numpy as np
for T in [1.9,2.0,2.1]:
    a2=np.load(f'/work/a2_T{T}.npy'); a4=np.load(f'/work/a4_T{T}.npy')
    print(T, a2.mean(), a2.std(), a4.mean(), a4.std(), a2[:5])
    # simple bootstrap over replicas
    rng=np.random.default_rng(0)
    Us=[]
    for b in range(300):
        idx=rng.integers(0,len(a2),len(a2))
        Us.append(1-a4[idx].mean()/(3*a2[idx].mean()**2))
    print(' bootstrap U=%.5f std=%.5f'%(1-a4.mean()/(3*a2.mean()**2), np.std(Us)))
