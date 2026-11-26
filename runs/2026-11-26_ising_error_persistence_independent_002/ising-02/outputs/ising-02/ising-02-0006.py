
import sys
sys.path.insert(0,'/work')
from ising import run, binder_jk
for T,seed in [(1.9,777),(2.2,778)]:
    m2,m4=run(T,1000,seed,6000,3000)
    U,err=binder_jk(m2,m4)
    print(T,U,err,flush=True)
