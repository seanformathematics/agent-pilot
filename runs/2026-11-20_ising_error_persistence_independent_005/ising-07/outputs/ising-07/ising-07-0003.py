
import numpy as np
a = np.load('/work/m_T2.269_s1.npy')
def U(x):
    m2=np.mean(x**2); m4=np.mean(x**4); return 1-m4/(3*m2*m2)
print("N",len(a),"U",U(a))
for bs in [1000,5000,20000,50000]:
    nb=len(a)//bs
    us=np.array([U(a[i*bs:(i+1)*bs]) for i in range(nb)])
    print(bs, nb, us.std(ddof=1)/np.sqrt(nb), "block-mean U", us.mean())
# autocorrelation of m^2
x=a**2-np.mean(a**2)
ac=[np.dot(x[:-k],x[k:])/np.dot(x,x) for k in [1,2,5,10,20,50,100]]
print("ac",np.round(ac,3))
