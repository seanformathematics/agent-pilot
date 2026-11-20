import subprocess, numpy as np, json
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
seeds={T:1000+i for i,T in enumerate(Ts)}
for T in Ts:
    r=subprocess.run(['python','/work/sw.py',str(T),'400000','2000',str(seeds[T]),f'/work/prod_{T}.npz'],capture_output=True,text=True,cwd='/work',timeout=550)
    print(T, r.stdout.strip(), r.stderr[-300:])
def U(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
res={}
for T in Ts:
    d=np.load(f'/work/prod_{T}.npz'); m2=d['m2']; m4=d['m4']
    u=U(m2,m4)
    nb=40; L=len(m2)//nb
    us=np.array([U(m2[i*L:(i+1)*L],m4[i*L:(i+1)*L]) for i in range(nb)])
    err=us.std(ddof=1)/np.sqrt(nb)
    # jackknife check
    jk=np.array([U(np.delete(m2.reshape(nb,L),i,0).ravel(),np.delete(m4.reshape(nb,L),i,0).ravel()) for i in range(nb)])
    errj=np.sqrt((nb-1)/nb*np.sum((jk-jk.mean())**2))
    res[T]=dict(U=float(u),err_bin=float(err),err_jk=float(errj),mean_m2=float(m2.mean()))
    print(T, res[T])
json.dump(res,open('/work/results.json','w'))