import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'

log_sigma_1_tildes = np.arange(-5, 5)
rho_tildes = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
log_sigma_1_hat_deltas = np.arange(-5, 5, step=.25)
params = [(s1, r, s1d) for s1 in log_sigma_1_tildes for r in rho_tildes for s1d in log_sigma_1_hat_deltas]
df = pd.DataFrame(params, columns=['log_sigma_1_tildes', 'rho_tildes', 'log_sigma_1_hat_deltas'])

df['log_sigma_1_hats'] = df['log_sigma_1_tildes'] + df['log_sigma_1_hat_deltas'] # = log(sigma_1_tilde * log_sigma_1_hat_deltas)
df['sigma_1_tildes'] = np.exp(df['log_sigma_1_tildes'])
df['sigma_1_hats'] = np.exp(df['log_sigma_1_hats'])

df['rho_hats'] = (df['rho_tildes'] * np.sqrt(df['sigma_1_hats'] / (df['rho_tildes'] ** 2 * 
                                              (df['sigma_1_hats'] - df['sigma_1_tildes']) + df['sigma_1_tildes'])))

matplotlib.rc('text', usetex=False)
g = sns.relplot(x='log_sigma_1_hat_deltas', y='rho_hats', hue='rho_tildes', 
                 errorbar=None, data=df, kind='line', legend='full')

g._legend.set_title(r'$r$')
g.set_xlabels('Variance Change \n'+r'(log($\hat{\sigma}^2_b$) - log($\tilde{\sigma}^2_b$))')
g.set_ylabels(r'$\hat{\rho}$')
g.savefig(fig_dir + "fig_3.pdf", transparent=True, bbox_inches='tight')