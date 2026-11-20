
import numpy as np
for T in [1.9,2.0]:
    d=np.load(f'/work/run_T{T}_a.npz'); s2=d['s2']; s4=d['s4']
    print(T, s2.shape, 'mean m2',s2.mean(),'mean m4',s4.mean())
    a2=s2.mean(0); a4=s4.mean(0)
    print(' per-replica std a2',a2.std(),' a4',a4.std())
    # error via block-of-blocks (time) for pooled ratio across replicas
    U=lambda S2,S4: 1-S4.mean()/(3*S2.mean()**2)
    nb=s2.shape[0]
    # jackknife over time blocks (pooled over replicas)
    loo=[U(np.delete(s2,k,0),np.delete(s4,k,0)) for k in range(nb)]
    loo=np.array(loo); n=nb
    print(' time-jackknife err', np.sqrt((n-1)/n*np.sum((loo-loo.mean())**2)), 'U',U(s2,s4))
