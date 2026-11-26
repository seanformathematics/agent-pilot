import numpy as np
from scipy.special import ellipk
def onsager_e(T):
    K=1/T
    k=2*np.sinh(2*K)/np.cosh(2*K)**2
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*ellipk(k*k))  # scipy ellipk takes m=k^2
sim={1.9:-1.815,2.0:-1.748,2.1:-1.662,2.2:-1.549,2.3:-1.40,2.4:-1.255}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    print(T, round(onsager_e(T),4), sim[T])