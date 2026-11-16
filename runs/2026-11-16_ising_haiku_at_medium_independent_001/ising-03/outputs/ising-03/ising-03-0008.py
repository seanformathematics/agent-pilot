
import sys, time, json; sys.path.insert(0,'/work')
from lib import run
for k,T in enumerate([1.5,1.6,1.7,1.8,1.9,2.0]):
    t=time.time()
    a2,a4=run(T,1000,20000,5000,1000+k)
    json.dump({'T':T,'m2':a2.tolist(),'m4':a4.tolist()},open(f'/work/res_{T}.json','w'))
    print('done',T,round(time.time()-t,1),flush=True)
