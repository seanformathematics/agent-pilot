
import numpy as np
NN=256
def U_of(m):
    m=m/NN
    return 1-np.mean(m**4)/(3*np.mean(m**2)**2)
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    m=np.load(f'/work/m_T{T}.npy').astype(np.float64)
    out=[]
    for nblk in [100,20]:
        L=len(m)//nblk; mm=m[:nblk*L]; blocks=mm.reshape(nblk,L)
        jk=np.array([U_of(np.delete(blocks,b,axis=0).ravel()) for b in range(nblk)])
        out.append(np.sqrt((nblk-1)/nblk*np.sum((jk-jk.mean())**2)))
    print(T, len(m), "U=%.6f"%U_of(m), "err100=%.2e err20=%.2e"%tuple(out))
