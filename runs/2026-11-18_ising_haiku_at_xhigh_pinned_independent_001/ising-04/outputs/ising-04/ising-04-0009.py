import sys, glob, json
sys.path.insert(0,'/work')
import numpy as np, ising
rows=[]
for f in sorted(glob.glob('/work/res/*.npz'), key=lambda s: float(s.split('/')[-1][1:5])):
    d=np.load(f)
    T=float(f.split('/')[-1][1:5]); seed=int(f.split('_s')[-1].split('.')[0])
    U,err=ising.binder_jk(d['A'],d['B'],int(d['K']))
    R=len(d['A']); K=int(d['K'])
    rows.append(dict(T=T,U=float(U),U_err=float(err),n=R*K,seed=seed))
    print(f"{T:.2f} U={U:.8f} err={err:.3e} n={R*K} seed={seed}")
json.dump(rows, open('/work/final_rows.json','w'), indent=1)
