
import sys, numpy as np
sys.path.insert(0,'/work')
import sim
temps = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
nsw, nb, dt = sim.run(seed=202, temps=temps, R=64, equil=1000, block=100, time_budget=200, out='/work/prod2.npz', L=16)
print("run2 sweeps", nsw, "blocks", nb)

def load(fn):
    d=np.load(fn); return d['m2'],d['m4'],d['temps'],int(d['R']),int(d['block'])
def Ufun(S2,S4,n):
    return 1-(S4.mean()/n)/(3*(S2.mean()/n)**2)
def est(list_of_files):
    # combine replicas across runs
    A2=[];A4=[];
    for fn in list_of_files:
        m2,m4,tt,R,bl=load(fn)
        nb=m2.shape[0]
        A2.append(m2.sum(0)); A4.append(m4.sum(0)); n=nb*bl
    S2=np.concatenate(A2); S4=np.concatenate(A4)  # shape (nrep_total, Rtot) pattern: we handle per T below
    return A2,A4,n,R,tt
files=['/work/prod1.npz','/work/prod2.npz']
data=[load(f) for f in files]
results={}
for k,T in enumerate(temps):
    row={}
    S2s=[];S4s=[];ns=[]
    for name,(m2,m4,tt,R,bl) in zip(['run1','run2'],data):
        sl=slice(k*R,(k+1)*R)
        S2=m2[:,sl].sum(0); S4=m4[:,sl].sum(0); n=m2.shape[0]*bl
        row[name]=(Ufun(S2,S4,n), )
        S2s.append(S2); S4s.append(S4); ns.append(n)
    # combined: replicas from both runs (same n across runs here)
    S2=np.concatenate(S2s); S4=np.concatenate(S4s); n=ns[0]
    Rt=len(S2)
    U=Ufun(S2,S4,n)
    jk=np.array([Ufun(np.delete(S2,i),np.delete(S4,i),n) for i in range(Rt)])
    err=np.sqrt((Rt-1)/Rt*np.sum((jk-jk.mean())**2))
    # run-to-run difference
    d12 = row['run1'][0]-row['run2'][0]
    results[T]=(U,err,row['run1'][0],row['run2'][0],n,Rt)
    print(f"T={T:.2f} run1={row['run1'][0]:.6f} run2={row['run2'][0]:.6f} diff={d12:+.6f}  combined U={U:.6f} err={err:.6f} (Nrep={Rt})")
np.save('/work/results.npy', np.array([[T,results[T][0],results[T][1]] for T in temps]))
