
import pickle, numpy as np
st=pickle.load(open('/work/state_T2.269_R1000_pilot.pkl','rb'))
acc=st['acc']; n=acc['n']; N=16
S2=acc['S2']/n; S4=acc['S4']/n
def U_of(idx):
    m2=S2[idx].mean(); m4=S4[idx].mean()
    return 1-m4/(3*m2**2)
R=len(S2)
U=U_of(np.arange(R))
jk=np.array([U_of(np.delete(np.arange(R),r)) for r in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print("U",U,"jk err",err)
# per-chunk check: split time? estimate via replicas halves
