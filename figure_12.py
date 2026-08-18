import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'

np.random.seed(10)
mean = [0, 0]
cov = [[1, .3], [.3, 1]] 
b, n = np.random.multivariate_normal(mean, cov, 200).T

df_true = pd.DataFrame({'Behavior': b, 'Neuroimaging': n, 
                        'Parameter': 'True', 'i': np.arange(200)})
b_hat = np.random.normal(loc=b, scale=1)
df_est = pd.DataFrame({'Behavior': b_hat, 'Neuroimaging': n, 
                       'Parameter': 'Estimate', 'i': np.arange(200)})
df = pd.concat([df_true, df_est])

g = sns.lmplot(x='Behavior', y='Neuroimaging', data=df[df['Parameter'] == 'True'][:25])
sns.regplot(x='Behavior', y='Neuroimaging', data=df[df['Parameter'] == 'True'][25:], 
            ax=g.axes[0, 0], scatter_kws={'alpha': .1}, line_kws={'alpha': .1})
g.set(xlim=(-2.5, 2), ylim=(-2.5, 2))
sns.despine()
g.savefig(fig_dir + "fig_1.pdf", transparent=True, bbox_inches='tight')

g = sns.lmplot(x='Behavior', y='Neuroimaging', hue='Parameter', data=df[df['i'] < 25], aspect=1, height=5)
g.set(xlim=(-2.5, 2), ylim=(-2.5, 2))
g.savefig(fig_dir + "fig_2.pdf", transparent=True, bbox_inches='tight')