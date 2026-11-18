
exec(open('/tmp/run_python/'+sorted(__import__('os').listdir('/tmp/run_python'))[0]).read().split('t=time.time()')[0])
import json
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=[]
for i,T in enumerate(Ts):
    seed=1000+i
    m=run(T,seed,300,3000)
    m2=(m**2).mean(0); m4=(m**4).mean(0)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=m2.sum(); S4=m4.sum()
    jk=1-((S4-m4)/(R-1))/(3*((S2-m2)/(R-1))**2)
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    # check first vs second half for equilibration
    h=len(m)//2
    Uh=[1-(m[a:b]**4).mean()/(3*(m[a:b]**2).mean()**2) for a,b in [(0,h),(h,None)]]
    res.append((T,U,err,m.size,seed)); print(T,U,err,Uh,flush=True)
json.dump(res,open('/work/res.json','w'))
