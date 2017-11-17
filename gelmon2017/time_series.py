import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()

import numpy as np
np.random.seed(2017)

from reda import ERT
obj = ERT()
import glob
for nr, filename in enumerate(sorted(
        # glob.glob('data2/pygimli_*.ohm')[0:100:10])):
        glob.glob('data2/pygimli_*.ohm'))):
    obj.import_bert(filename, timestep=nr)

# break

# fig, ax = plt.subplots()
# ax.plot(obj.data['R'].values, '.')
# ax.plot(obj.data['R'].values + noise_var * noise, '.')
# fig.savefig('noise.png', dpi=400)


obj.compute_K_analytical(spacing=1)

table = obj.data.head(5).to_latex(escape=False)
print(table)

import IPython
IPython.embed()


def plot_quadpole_evolution(data, quadpole, cols, rolling=False, ax=None):
    subquery = data.query(
        'A == {0} and B == {1} and M == {2} and N == {3}'.format(*quadpole)
    )

    rhoa = subquery['rho_a'].values
    rhoa[30] = 300
    subquery['rho_a'] = rhoa

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 6 / 2.54))

    if rolling:
        # rolling mean
        rolling_m = subquery.rolling(13, center=True, min_periods=1).median()
        ax.plot(
            rolling_m['timestep'].values,
            rolling_m['rho_a'].values,
            '-',
            label='rolling median',
        )
        ax.plot(subquery['timestep'], subquery[cols], '.',
                color='blue', label='valid data')

        threshold = 0.05
        ax.fill_between(
            rolling_m['timestep'].values,
            rolling_m['rho_a'].values * (1 - threshold),
            rolling_m['rho_a'].values * (1 + threshold),
            alpha=0.4,
            color='blue',
            label='5\% confidence region',
        )

        # import IPython
        # IPython.embed()
        # find all values that deviate by more than X percent from the
        # rolling_m
        bad_values = (
            np.abs(np.abs(
                subquery['rho_a'].values - rolling_m['rho_a'].values
            ) / rolling_m['rho_a'].values) > threshold
        )

        bad = subquery.loc[bad_values]
        ax.plot(
            bad['timestep'].values,
            bad['rho_a'].values,
            '.',
            # s=15,
            color='r',
            label='discarded data',
        )

        # ax2 = ax.twinx()
        # rolling_std = subquery['rho_a'].rolling(13).std()
        # # # import IPython
        # # # IPython.embed()

        # ax2.plot(
        #     subquery['timestep'].values,
        #     rolling_std.values,
        #     color='r',

        # )
    ax.legend(
        loc='upper center',
        fontsize=6
    )
    # ax.set_xlim(10, 20)

    ax.set_ylabel(r'$\rho~[\Omega m]$')
    ax.set_xlabel('timestep')
    ax.set_xlim(3, 96)
    return fig, ax


quadpole = [10, 11, 15, 14]
# fig, ax = plot_quadpole_evolution(obj.data, quadpole, 'rho_a')
# add data noise to R
N = obj.data.shape[0]
noise_var = np.abs(obj.data['R'].values) * 0.05
noise = np.random.randn(N)
data_new = obj.data['R'].values + noise_var * noise

obj.data['R'] = data_new

obj.compute_K_analytical(spacing=1)

fig, ax = plot_quadpole_evolution(obj.data, quadpole, 'rho_a', rolling=True)
# ax.set_title(quadpole)

# std = obj.data['rho_a'].rolling(5).std()
# print(std)
# import IPython
# IPython.embed()

fig.tight_layout()
fig.savefig('plot_ts.png', dpi=300)

# import IPython
# IPython.embed()
