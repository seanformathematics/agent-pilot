
import numpy as np, time
exec(open('/work/ising.py').read())
T=2.3; seed=12345
sw, cnt, m2r, m4r, ts = run(T, R=2000, seed=seed, n_therm=1000, t_budget=400)
np.savez(f'/work/res_T{T}_s{seed}.npz', m2r=m2r, m4r=m4r, ts=ts, sweeps=sw, cnt=cnt)
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
U=U_of(m2r,m4r)
R=len(m2r)
jk=np.array([U_of(np.delete(m2r,i),np.delete(m4r,i)) for i in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print("sweeps",sw,"cnt",cnt,"U",U,"jk err",err)
# time-series check: first vs second half of measurement
h=len(ts)//2
for a,b in [(0,h),(h,len(ts))]:
    m2=ts[a:b,0].mean(); m4=ts[a:b,1].mean()
    print("segment",a,b,"U_seg",1-m4/(3*m2*m2))
