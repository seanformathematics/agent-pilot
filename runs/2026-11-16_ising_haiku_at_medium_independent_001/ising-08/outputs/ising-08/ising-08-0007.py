import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, json, time
sys.path.insert(0,'/work')
from ising import run
from multiprocessing import Pool
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
args=[(T, 1000+i, 1000, 4000, 8000) for i,T in enumerate(Ts)]
if __name__=="__main__":
    t0=time.time()
    with Pool(8) as p:
        for T,m2,m4 in p.imap_unordered(run,args):
            json.dump({'T':T,'m2':m2,'m4':m4}, open(f'/work/res_{T}.json','w'))
            print('done',T,round(time.time()-t0),flush=True)
