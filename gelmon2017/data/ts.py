import numpy as np
import pylab as plt
from reda import ERT
obj = ERT()
import glob
for nr, filename in enumerate(sorted(glob.glob('pygimli_*.ohm'))):
    obj.import_bert_ohm(filename, timestep=nr)

obj.compute_K_analytical(spacing=1)


def plot_quadpole_evolution(data, quadpole, cols):
    subquery = data.query(
        'A == {0} and B == {1} and M == {2} and N == {3}'.format(*quadpole)
    )
    fig, ax = plt.subplots(1, 1)
    ax.plot(subquery['timestep'], subquery[cols])
    return fig, ax

fig, ax = plot_quadpole_evolution(obj.data, [1, 2, 4, 3], 'rho_a')

plt.savefig('t1.png', dpi=300)
