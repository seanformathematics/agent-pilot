
exec(open('/tmp/run_python/'+sorted(__import__('os').listdir('/tmp/run_python'))[0]).read().split('t0=time')[0])
import json,os
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=json.load(open('/work/res.json')) if os.path.exists('/work/res.json') else {}
t0=time.time()
for k,T in enumerate(Ts):
    if str(T) in res: continue
    if time.time()-t0>450: break
    U,e,n=run(T,R=1000,neq=3000,nmeas=6000,seed=1000+k)
    res[str(T)]=[U,e,int(n),1000+k]; json.dump(res,open('/work/res.json','w'))
    print(T,U,e,n,time.time()-t0,flush=True)
