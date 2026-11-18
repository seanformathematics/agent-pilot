
exec(open('/tmp/run_python/'+sorted(__import__('os').listdir('/tmp/run_python'))[0]).read().split('t0=time.time()')[0])
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=[]
for k,T in enumerate(Ts):
    U,e,n=run(T,1000+k,1000,10000); res.append((T,U,e,n)); print(T,U,e,n,flush=True)
json.dump(res,open('/work/res.json','w'))
