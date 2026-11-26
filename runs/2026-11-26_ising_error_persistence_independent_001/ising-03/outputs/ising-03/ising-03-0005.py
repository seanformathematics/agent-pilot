
import json, numpy as np
def U(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
out={}
for T in ['1.9','2.0','2.1','2.2','2.3','2.4']:
    chains=[np.array(json.load(open(f'/work/res_{T}_{sd}.json'))['bins']) for sd in [301,302,303]]
    per=[U(c[:,0],c[:,1]) for c in chains]
    allb=np.vstack(chains)
    m2=allb[:,0]; m4=allb[:,1]
    nb=len(allb)
    Us=np.array([U(np.delete(m2,k),np.delete(m4,k)) for k in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
    # chain-level jackknife
    Uc=[]
    for k in range(3):
        idx=[j for j in range(3) if j!=k]
        c=np.vstack([chains[j] for j in idx])
        Uc.append(U(c[:,0],c[:,1]))
    Uc=np.array(Uc); errc=np.sqrt(2/3*np.sum((Uc-Uc.mean())**2))
    print(T, 'U=%.5f'%U(m2,m4), 'jack_err=%.5f'%err, 'chain-jk err=%.5f'%errc, 'per-chain', np.round(per,5))
