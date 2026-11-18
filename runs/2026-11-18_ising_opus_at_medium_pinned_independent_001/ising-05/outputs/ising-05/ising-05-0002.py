
exec(open('/tmp/run_python/'+sorted(__import__('os').listdir('/tmp/run_python'))[0]).read().split("t=time.time()")[0])
import json
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
for i,T in enumerate(Ts):
    ms=run(T,200000,1000+i)
    U,e=U_jk(ms)
    # first-half vs second-half check for equilibration
    U1,_=U_jk(ms[:100000],50); U2,_=U_jk(ms[100000:],50)
    res[T]=(U,e,len(ms)); print(T,U,e,U1,U2,flush=True)
json.dump(res,open('/work/res.json','w'))
