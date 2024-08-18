from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

def prob_bi_greater_than_or_equal_two(n, zeta):
    return 1 - binom.pmf(0, n, zeta) - binom.pmf(1, n, zeta)

def prob_bim_greater_than_or_equal_two(n, zeta):
    return (1 - binom.pmf(0, n, zeta))**2

def avg_g20(ns, zeta, lamb, nframes=500000):
    B = nframes * lamb * zeta
    avgG20 = nframes * prob_bi_greater_than_or_equal_two(ns, zeta)
    avgG2m = nframes * prob_bim_greater_than_or_equal_two(ns, zeta)
    return avgG20, avgG2m

def get_ratio_vN(N_values,zeta,lambda_val,nframes):
    G20_values = []
    G2m_values = []
    for N in N_values:
        avgG20, avgG2m = avg_g20(N, zeta, lambda_val, nframes)
        G20_values.append(avgG20)
        G2m_values.append(avgG2m)
    G20_values = np.array(G20_values)
    G2m_values = np.array(G2m_values)
    return G20_values,G2m_values


lamb = 0.0
nframes = 500000
fig,ax=plt.subplots(1,2,figsize=(6,3))

N_values = np.arange(1,10,1)
zeta1 = 1e-4
G20_values1,G2m_values1 = get_ratio_vN(N_values,zeta1,lamb,nframes)
zeta2 = 1e-3
G20_values2,G2m_values2 = get_ratio_vN(N_values,zeta2,lamb,nframes)
zeta3 = 1e-2
G20_values3,G2m_values3 = get_ratio_vN(N_values,zeta3,lamb,nframes)

ax[0].scatter(N_values, G20_values1/G2m_values1,edgecolor='red',label=r'$\zeta=$'+f'{zeta1}',facecolor='white',s=10)
ax[0].scatter(N_values, G20_values2/G2m_values2,edgecolor='blue',label=r'$\zeta=$'+f'{zeta2}',facecolor='white',s=10)
ax[0].scatter(N_values, G20_values3/G2m_values3,edgecolor='black',label=r'$\zeta=$'+f'{zeta3}',facecolor='white',s=10)
ax[0].legend(frameon=False)
    
N_values = np.arange(1,1000,20)
zeta1 = 1e-4
G20_values1,G2m_values1 = get_ratio_vN(N_values,zeta1,lamb,nframes)
zeta2 = 1e-3
G20_values2,G2m_values2 = get_ratio_vN(N_values,zeta2,lamb,nframes)
zeta3 = 1e-2
G20_values3,G2m_values3 = get_ratio_vN(N_values,zeta3,lamb,nframes)

ax[1].scatter(N_values, G20_values1/G2m_values1,edgecolor='red',label=r'$\zeta=$'+f'{zeta1}',facecolor='white',s=10)
ax[1].scatter(N_values, G20_values2/G2m_values2,edgecolor='blue',label=r'$\zeta=$'+f'{zeta2}',facecolor='white',s=10)
ax[1].scatter(N_values, G20_values3/G2m_values3,edgecolor='black',label=r'$\zeta=$'+f'{zeta3}',facecolor='white',s=10)


for axi in ax.ravel():
    axi.set_xlabel('N')
    axi.set_ylabel(r'$\langle G^{2}(0)\rangle/\langle G^{2}(m)\rangle$')
    axi.set_title(r'$B=0$')
    axi.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/g20g2m.png',dpi=300)
plt.show()


