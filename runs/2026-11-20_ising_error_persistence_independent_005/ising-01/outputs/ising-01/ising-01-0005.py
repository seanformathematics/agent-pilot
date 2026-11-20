import json, numpy as np
out=json.load(open('/work/res.json'))
L=256
def U_of(B):
    n=B[:,0].sum(); m2=B[:,1].sum()/n; m4=B[:,2].sum()/n
    return 1-(m4/L**4)/(3*(m2/L**2)**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    B=np.array(out[f"{T}_101"],dtype=float)
    nb=len(B)
    U=U_of(B)
    # jackknife
    jk=np.array([U_of(np.delete(B,i,axis=0)) for i in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((jk-jk.mean())**2))
    # halves
    h1=U_of(B[:nb//2]); h2=U_of(B[nb//2:])
    print(f"T={T} U={U:.5f} err_jk={err:.5f} halves {h1:.4f} {h2:.4f} <m2>={B[:,1].sum()/B[:,0].sum()/L**2:.4f}")