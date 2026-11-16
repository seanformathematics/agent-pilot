import os,sys,json,time
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/work')
from ising import run
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
for i,T in enumerate(Ts):
    _,m2,m4=run((T,3000+i,1000,3000,6000))
    json.dump({'T':T,'m2':m2,'m4':m4},open(f'/work/res_{T}.json','w'))
    print('done',T,round(time.time()-t0),flush=True)