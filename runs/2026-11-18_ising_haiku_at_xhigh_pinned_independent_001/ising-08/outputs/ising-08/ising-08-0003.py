
import numpy as np
d = np.load('/work/pilot.npz')
m2 = d['m2']; m4 = d['m4']; am = d['am']; temps = d['temps']; R = int(d['R']); block=int(d['block'])
nb, Rtot = m2.shape
print("blocks", nb, "Rtot", Rtot)
nT = len(temps)
def U_from(S2, S4, n):
    # S2,S4 arrays over replicas (sums), n = number of samples per replica
    m2b = S2.mean(); m4b = S4.mean()
    return 1 - (m4b/n)/(3*(m2b/n)**2)
res = []
for k,T in enumerate(temps):
    sl = slice(k*R,(k+1)*R)
    S2 = m2[:,sl].sum(axis=0); S4 = m4[:,sl].sum(axis=0)
    n = nb*block
    U = U_from(S2,S4,n)
    # jackknife over replicas
    jk = np.empty(R)
    for i in range(R):
        mask = np.ones(R,bool); mask[i]=False
        jk[i] = U_from(S2[mask],S4[mask],n)
    err = np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    res.append((T,U,err))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f}  <|m|>={am[:,sl].mean()/block/1:.4f}" )
# drift check: block-averaged U over all replicas vs block index, for a few temps
for k in [0,5,9,13]:
    sl = slice(k*R,(k+1)*R)
    Ub = []
    for b in range(0,nb,50):
        seg = slice(b,b+50)
        S2 = m2[seg,sl].sum(); S4 = m4[seg,sl].sum()
        Ub.append(1 - (S4/(50*R*block))/(3*(S2/(50*R*block))**2))
    print("T",temps[k], np.round(Ub,4))
