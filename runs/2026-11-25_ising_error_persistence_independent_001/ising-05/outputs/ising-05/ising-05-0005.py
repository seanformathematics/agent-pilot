
import numpy as np
d=np.load('/work/state_T2.269.npz')
A=d['A'];B=d['B'];C=d['C']
R,nb=A.shape
print(R,nb)
def U_of(A,B):
    a=A.mean(); b=B.mean()
    return 1-b/(3*a*a)
def jk(A,B):
    R=A.shape[0]
    Ar=A.mean(1); Br=B.mean(1)  # per replica
    sa=Ar.sum(); sb=Br.sum()
    Us=np.empty(R)
    for r in range(R):
        a=(sa-Ar[r])/(R-1); b=(sb-Br[r])/(R-1)
        Us[r]=1-b/(3*a*a)
    U=1-(sb/R)/(3*(sa/R)**2)
    err=np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
    return U,err
U,e=jk(A,B)
print('U',U,'err',e)
# stationarity: mean |m| per block over time
Cm=C.mean(0)
print('blockwise <|m|> first 20 blocks', np.round(Cm[:20],4))
print('blockwise <|m|> last 20', np.round(Cm[-20:],4))
# U by thirds
for k in range(4):
    sl=slice(k*nb//4,(k+1)*nb//4)
    print(k, jk(A[:,sl],B[:,sl]))
