
import sys, time, json; sys.path.insert(0,'/work')
from lib import run
for k,T in enumerate([2.4,2.5,2.6,2.7,2.8]):
    t=time.time()
    a2,a4=run(T,1000,20000,5000,3000+k)
    json.dump({'T':T,'m2':a2.tolist(),'m4':a4.tolist()},open(f'/work/res_{T}.json','w'))
    print('done',T,round(time.time()-t,1),flush=True)
