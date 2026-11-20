import numpy as np
def U_of(b2,b4):
    return 1-b4.mean()/(3*b2.mean()**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/T{T}.npz"); b2=d["m2"]; b4=d["m4"]; n=len(b2)
    U=U_of(b2,b4)
    # jackknife
    S2=b2.sum(); S4=b4.sum()
    jk=np.empty(n)
    for i in range(n):
        m2=(S2-b2[i])/(n-1); m4=(S4-b4[i])/(n-1)
        jk[i]=1-m4/(3*m2**2)
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    # block-of-blocks check: 20 superblocks
    sb=20; k=n//sb
    sU=[U_of(b2[j*k:(j+1)*k],b4[j*k:(j+1)*k]) for j in range(sb)]
    print(T, "U=%.5f"%U, "jk_err=%.5f"%err, "superblock_err=%.5f"%(np.std(sU,ddof=1)/np.sqrt(sb)), "n_blocks",n)
