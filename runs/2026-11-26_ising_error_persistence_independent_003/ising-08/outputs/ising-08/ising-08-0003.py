
import sys; sys.path.insert(0,'/work')
from ising import run
import statistics as st
for T in [1.0,2.0,2.3,5.0,10.0]:
    m2,m4=run(16,T,7,2000,20000)
    a=sum(m2)/len(m2); b=sum(m4)/len(m4)
    print(T, a, b, 1-b/(3*a*a))
