
import numpy as np, time, json
exec(open('/work/ising.py').read())
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
out={}
for k,T in enumerate([1.9,2.0,2.1,2.2,2.3,2.4]):
    seed=1000+k
    sw, cnt, m2r, m4r, ts = run(T, R=2000, seed=seed, n_therm=1000, t_budget=55)
    np.savez(f'/work/res_T{T}_s{seed}.npz', m2r=m2r, m4r=m4r, ts=ts, sweeps=sw, cnt=cnt)
    U=U_of(m2r,m4r); R=len(m2r)
    jk=np.array([U_of(np.delete(m2r,i),np.delete(m4r,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    h=len(ts)//2
    segs=[1-ts[a:b,1].mean()/(3*ts[a:b,0].mean()**2) for a,b in [(0,h),(h,len(ts))]]
    out[T]=dict(seed=seed,U=U,err=err,sweeps=sw,cnt=cnt,segs=segs)
    print(f"T={T} seed={seed} sweeps={sw} meas={cnt} U={U:.6f} err={err:.2e} halves={segs[0]:.5f},{segs[1]:.5f}", flush=True)
json.dump({str(k):v for k,v in out.items()}, open('/work/summary.json','w'), indent=1)
