import json, numpy as np
A=json.load(open('/work/resA.json')); B=json.load(open('/work/resB.json'))
rows=[]
for key in sorted(A.keys(), key=float):
    a=A[key]; b=B[key]
    U=(a['U']+b['U'])/2
    err=np.sqrt(a['err']**2+b['err']**2)/2
    z=(a['U']-b['U'])/np.sqrt(a['err']**2+b['err']**2)
    rows.append((float(key),U,err,a['U'],b['U'],z,a['seed'],b['seed']))
    print(f"T={float(key):.2f}  U={U:.7f}  err={err:.2e}  A={a['U']:.7f}({a['err']:.1e})  B={b['U']:.7f}({b['err']:.1e})  z={z:+.2f}")
print("max err", max(r[2] for r in rows))
print("seeds A", [A[k]['seed'] for k in sorted(A,key=float)])
print("seeds B", [B[k]['seed'] for k in sorted(B,key=float)])
print("chi2 of z:", np.mean([r[5]**2 for r in rows]))
json.dump([dict(T=r[0],U=r[1],err=r[2]) for r in rows], open('/work/final.json','w'))
