
import sys, time, numpy as np, json
sys.path.insert(0,'/work')
import wolff
def U_of(b2,b4): return 1-b4.mean()/(3*b2.mean()**2)
def jk(b2,b4,nb=50):
    n=len(b2); idx=np.arange(n)
    groups=np.array_split(idx,nb)
    Us=[]
    for g in groups:
        mask=np.ones(n,bool); mask[g]=False
        Us.append(U_of(b2[mask],b4[mask]))
    Us=np.array(Us)
    return float(np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2)))
res={}
for T,sd in [(2.2,301),(2.3,302),(2.4,303)]:
    t=time.time()
    s,M,b2,b4 = wolff.run(T, sd, 600000, 100)
    np.savez(f'/work/blocks_T{T}.npz', b2=b2, b4=b4)
    h=len(b2)//2
    res[T]=dict(U=U_of(b2,b4), err=jk(b2,b4), U_h1=U_of(b2[:h],b4[:h]), U_h2=U_of(b2[h:],b4[h:]), m2=b2.mean(), time=time.time()-t)
    print(T,res[T],flush=True)
json.dump({str(k):v for k,v in res.items()}, open('/work/res_b.json','w'))
