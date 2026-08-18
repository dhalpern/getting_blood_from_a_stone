import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import itertools
import getting_blood_from_a_stone_funcs as gbfs

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'

ns = [15, 20, 25, 50, 100]
Ts = [25, 50, 100, 200]
CNRs = [.05, .1, .15, .2] 
ICCs = [(2 / 3)]
ICCs = [.397]
rhos = np.linspace(0.01, 1, num=200)
prods = list(itertools.product(ns, rhos, Ts, CNRs, ICCs))

power_res_dicts = []
for n, rho, T, CNR, ICC in prods:
    power = gbfs.t_test_power(gbfs.group_t(rho, ICC, CNR, T, n), n)
    res_dict = {'n': n,
               'T': T,
               'CNR': CNR,
               'ICC': ICC,
               'rho': rho, 
               'power': power}
    power_res_dicts.append(res_dict)
power_df = pd.DataFrame(power_res_dicts)

g = sns.relplot(x='rho', y='power', row='CNR', col='T',
            hue='n', legend='full',  
            kind='line', data=power_df)
g._legend.set_title('Neuroimaging Data ($N_{nb}$)')
sns.move_legend(g, "center left", bbox_to_anchor=(.9, 0.5))
g.set_axis_labels("", "")
g.fig.supylabel(r'Power')
g.fig.supxlabel(r'Estimation Quality $\rho(\mathbf{c}_i, \hat{\mathbf{c}}_i)$')
g.savefig(fig_dir + "/fig_8.pdf", transparent=True, bbox_inches='tight')

n_extra = 1

achievable_power_df = power_df.query('n <= 50 and CNR == .15')

res_dicts = []
for n, T, CNR, ICC, rho, power in achievable_power_df.itertuples(index=False):
    if gbfs.solve_group_t((1 - rho), rho, ICC, CNR, T, n, n_extra) > 0:
        rho_diff = sp.optimize.root_scalar(gbfs.solve_group_t, bracket=[0, (1 - rho)],
                       args=(rho, ICC, CNR, T, n, n_extra)).root
        res_dict = {
            'rho': rho,
            'n': n,
            'T': T,
            'CNR': CNR,
            'ICC': ICC,
            'rho_diff': rho_diff}
        res_dicts.append(res_dict)
    else:
        continue
res_df = pd.DataFrame(res_dicts)

g = sns.relplot(x='rho', y='rho_diff', col='n', 
            hue='T', legend='full',
            kind='line', data=res_df, 
                facet_kws={'xlim': [0, 1]})
g.set_titles(r'Neuroimaging Data ($N_{{xy}}$) = {col_name}')
g._legend.set_title('Trials (T)')
g.set_axis_labels("", "")
g.fig.supylabel('Estimation Improvement \n to Match ' + r'$N_{xy} + 1$ ($\delta_{\rho}$)', ha="center", x=0)
g.fig.supxlabel(r'Neuroimaging Data Estimation Quality ($\rho(\mathbf{c}_i, \hat{\mathbf{c}}_i)$)')
g.savefig(fig_dir + "/fig_9.pdf", transparent=True, bbox_inches='tight')