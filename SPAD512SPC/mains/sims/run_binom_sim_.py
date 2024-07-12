from SPAD512SPC.generators import SPAD2D_Ring
import matplotlib.pyplot as plt

config = {
'nx': 10,
'ny': 10,
'sigma': 0.55,
'particles': 3,
'lamb': 0.0003,
'zeta': 0.01,
'nt': 100000,
}
generator = SPAD2D_Ring(config)
x = generator.generate(ring_radius=1.0,show=True)
print(x.shape)
plt.show()
