import os,time,glob,sys
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/work')
print(glob.glob('/work/res_*'))
from ising import run
t=time.time(); run((2.3,5,1000,100,100)); print('200 sweeps', time.time()-t)
t=time.time(); run((1.5,5,1000,100,100)); print('200 sweeps T1.5', time.time()-t)