import numpy as np
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m):
    m2=m*m
    return 1-np.mean(m2*m2)/(3*np.mean(m2)**2)
for T in temps:
    m=np.load(f'/work/data/m_T{T}.npy')
    res=[]
    for nb in [20,50,100,200]:
        L=len(m)//nb
        blk=m[:L*nb].reshape(nb,L)
        Us=np.array([U_of(np.delete(blk,k,axis=0).ravel()) for k in range(nb)])
        err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
        res.append((nb,round(err,5)))
    # independent-seed-like check: split into 10 chunks
    chunks=np.array_split(m,10)
    Uc=np.array([U_of(c) for c in chunks])
    print(T, "U=%.5f"%U_of(m), "jackknife err (nblocks: err)",res, "chunk-based err of 10 chunks: %.5f"%(Uc.std(ddof=1)/np.sqrt(10)))
