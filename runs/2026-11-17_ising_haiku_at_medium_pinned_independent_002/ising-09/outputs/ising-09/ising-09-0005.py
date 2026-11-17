
import numpy as np, glob
NN=256
def U_of(m):
    m=m/NN
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2**2)
res={}
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    m=np.load(f'/work/m_T{T}.npy').astype(np.float64)
    nblk=100
    L=len(m)//nblk
    m=m[:nblk*L]
    U=U_of(m)
    blocks=m.reshape(nblk,L)
    jk=[]
    for b in range(nblk):
        mm=np.delete(blocks,b,axis=0).ravel()
        jk.append(U_of(mm))
    jk=np.array(jk)
    err=np.sqrt((nblk-1)/nblk*np.sum((jk-jk.mean())**2))
    # autocorr check of block-level: compare err with 20 blocks
    res[T]=(U,err,len(m))
    print(T, len(m), round(U,5), round(err,5), "mean|m|", round(np.mean(np.abs(m))/NN,4))
