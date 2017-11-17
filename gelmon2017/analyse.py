import numpy as np
from reda import ERT
obj = ERT()
import glob

filenames = sorted(glob.glob('data/pygimli_*.ohm'))
for nr, filename in enumerate(filenames):
    obj.import_bert(filename, timestep=nr)

obj.compute_K_analytical(spacing=1)

# import reda.utils.norrec as NR
# NR.assign_norrec_to_df(obj.data)

import reda.plotters.histograms as RH
results = RH.plot_histograms_extra_dims(
    obj,
    keys=['rho_a'],
    extra_dims=['timestep', ],
    Nx=5,
)
results.tight_layout()
results.savefig('histogram.png', dpi=300)

exit()
import IPython
IPython.embed()
exit()
# import IPython
# IPython.embed()
for key, item in results.items():
    item['all'].savefig('hist_{0}.png'.format(key), dpi=300)

import reda.plotters.pseudoplots as PS
# PS.plot_pseudosection_type2(obj, 'R')
# fig, ax = PS.plot_pseudosection_type2(obj, 'R')
# fig.savefig('test.png', dpi=300)


def fancyfy(axes, N):
    for ax in axes[0:-1, :].flat:
        ax.set_xlabel('')
    for ax in axes[:, 1:].flat:
        ax.set_ylabel('')


g = obj.data.groupby('timestep')
N = len(g.groups.keys())
nrx = 5
nry = int(np.ceil(N / nrx))
sizex = nrx * 3
sizey = nry * 3 - 1
fig, axes = PS.plt.subplots(
    nry, nrx,
    sharex=True, sharey=True,
    figsize=(sizex, sizey),
)

cbs = []
for ax, (name, group) in zip(axes.flat, g):
    fig1, axes1, cb1 = PS.plot_pseudosection_type2(
        group, 'rho_a', ax=ax, log10=False,
        cbmin=50, cbmax=300,
    )
    cbs.append(cb1)
    ax.set_title('timestep: {0}'.format(name))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_aspect('equal')

for cb in np.array(cbs).reshape(axes.shape)[:, 0:-1].flat:
    cb.ax.set_visible(False)

# import IPython
# IPython.embed()

fancyfy(axes, N)
fig.tight_layout()
fig.savefig('pseudosections.png', dpi=300)
