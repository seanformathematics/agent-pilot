import json, numpy as np
out=json.load(open('/work/res.json'))
L=256
def U_of(B):
    n=B[:,0].sum(); m2=B[:,1].sum()/n; m4=B[:,2].sum()/n
    return 1-(m4/L**4)/(3*(m2/L**2)**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    B=np.array(out[f"{T}_101"]+out[f"{T}_202"],float); n=len(B)
    U=U_of(B)
    jk=np.array([U_of(np.delete(B,i,axis=0)) for i in range(n)])
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    print(T, f"{U:.5f}", f"{err:.5f}", int(B[:,0].sum()))