import os,sys,json,time
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/work')
from ising import run
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
t0=time.time()
for i,T in enumerate(Ts):
    _,m2,m4=run((T,2000+i,1000,3000,6000))
    json.dump({'T':T,'m2':m2,'m4':m4},open(f'/work/res_{T}.json','w'))
    print('done',T,round(time.time()-t0),flush=True)