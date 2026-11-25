
import sys,time
sys.path.insert(0,'/work')
from prod import *
out=[]
for T,seed in [(1.9,21),(2.0,31),(2.1,41)]:
    t=time.time()
    s2,s4=production(T,seed,nmeas=20000)
    U,e=ujack(s2,s4,20000)
    print(T,"U=",U,"err=",e,"time",time.time()-t, flush=True)
