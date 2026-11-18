
import numpy as np
files=['/work/prod1.npz','/work/prod2.npz']
data=[]
for f in files:
    d=np.load(f); data.append((d['m2'],d['m4'],d['am'],int(d['R']),int(d['block']),d['temps']))
temps=data[0][5]
final=[]
print("T, U_run1, U_run2, U_combined, err_combined, |diff|/sqrt(e1^2+e2^2)")
for k,T in enumerate(temps):
    S2=[];S4=[];NS=[]
    for m2,m4,am,R,bl,tt in data:
        sl=slice(k*R,(k+1)*R)
        S2.append(m2[:,sl].sum(0)); S4.append(m4[:,sl].sum(0)); NS.append(m2.shape[0]*bl*np.ones(R))
    S2=np.concatenate(S2); S4=np.concatenate(S4); NS=np.concatenate(NS)
    def U(idx):
        return 1-(S4[idx].sum()/NS[idx].sum())/(3*(S2[idx].sum()/NS[idx].sum())**2)
    allidx=np.arange(len(S2))
    Ut=U(allidx)
    Rt=len(S2)
    jk=np.array([U(np.delete(allidx,i)) for i in range(Rt)])
    err=np.sqrt((Rt-1)/Rt*np.sum((jk-jk.mean())**2))
    # per run
    ests=[]
    for r in range(2):
        R=data[r][3]; idx=np.arange(r*64,(r+1)*64)
        Ur=U(idx)
        jkr=np.array([U(np.delete(idx,i)) for i in range(64)])
        er=np.sqrt(63/64*np.sum((jkr-jkr.mean())**2))
        ests.append((Ur,er))
    z=(ests[0][0]-ests[1][0])/np.sqrt(ests[0][1]**2+ests[1][1]**2+1e-300)
    print(f"{T:.2f}  {ests[0][0]:.6f}±{ests[0][1]:.6f}  {ests[1][0]:.6f}±{ests[1][1]:.6f}  U={Ut:.6f} err={err:.6f}  z={z:+.2f}")
    final.append((T,Ut,err))
np.save('/work/final.npy',np.array(final))
