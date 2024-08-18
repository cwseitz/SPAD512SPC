import numpy as np
import matplotlib.pyplot as plt
from binom import BinomialPoissonSimulation

zeta = 0.01
ns = [1,3,5]
postns = np.arange(1,10,1)
nframes = 100000
lambs = [0.0, 0.0075, 0.01]

fig, ax = plt.subplots(len(lambs), len(ns), figsize=(7, 7), sharex=True, sharey=True)

column_titles = [r'$N_{true}$ ' + f'= {n}' for n in ns]
for m, col_title in enumerate(column_titles):
    ax[0,m].set_title(col_title)
    ax[-1,m].set_xlabel('N')

for l, lamb in enumerate(lambs):
    sim = BinomialPoissonSimulation()
    for m, this_n in enumerate(ns):
        x = sim.sample(this_n, zeta, nframes, lamb=lamb)
        avg_post = sim.post(x, postns, lamb=lamb, zeta_mean=zeta, num_samples=1000)
        ax[l, m].bar(postns, avg_post, color='white', edgecolor='black')
        ax[l,m].set_xticks([])
        ax[l,m].set_yticks([])
        ax[l,m].spines[['right', 'top']].set_visible(False)

even_xticks = [x for x in postns if x % 2 == 0]
for m in range(len(ns)):
    ax[len(lambs) - 1, m].set_xticks(even_xticks)

row_labels = [f'λ = {lamb}' for lamb in lambs]
for l, row_label in enumerate(row_labels):
    ax[l, 0].set_ylabel(row_label, rotation=0, labelpad=40, va='center')

plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/PoissonBinomialPost-1.png', dpi=300)
plt.show()

