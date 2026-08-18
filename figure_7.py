import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'
sim_dir = 'sims/'

summary_df_corr_means = pd.read_csv(sim_dir+'fig_7_summary_df_corr_means.csv')

g = sns.relplot(x="n_extra_beh", y="corr", hue="n_mri", legend='full', 
            facet_kws={'sharey': False, 'margin_titles': True},
            kind="line", estimator=None, 
            data=summary_df_corr_means)
g._legend.set_title('Neuroimaging Data ($N_{nb}$)')
sns.move_legend(g, "center left", bbox_to_anchor=(.75, 0.5))
g.set_xlabels(r'Behavioral Data ($N_{x}$)')
g.set_ylabels(r'Estimation Quality ($\rho({\theta, \hat{\theta}})$)')
g.savefig(fig_dir + "fig_7.pdf", transparent=True, bbox_inches='tight')