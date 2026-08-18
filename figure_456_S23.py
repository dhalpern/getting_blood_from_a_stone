import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'
sim_dir = 'sims/'

rhos = [.1, .3, .5, .7, .9]
n_mris = [25, 50, 100, 500, 1000]
n_sims = 1000000
summary_df_means_list = []
for rho in rhos:
    print('rho', rho)
    for n_mri in n_mris:
        summary_df_means_list.append(
            pd.read_csv(sim_dir+'rho'+str(rho)+'_n_mri'+str(n_mri)+'_n_sims'+str(n_sims)+'_fig4_summary_df_means.csv'))

summary_df_means_all = pd.concat(summary_df_means_list)

summary_df_means = summary_df_means_all.query('method != "anderson_rho_ddof0"')

summary_df_means['rho_rmse'] = np.sqrt(summary_df_means['rho_sq_error'])

# no need to simulate 0 behavioral data
summary_df_pearson_neb0 = summary_df_means.query('method != "anderson_rho" and n_extra_beh == 25')
summary_df_pearson_neb0['n_extra_beh'] = 0
summary_df_anderson_neb0 = summary_df_pearson_neb0.query('method == "corr_r"')
summary_df_anderson_neb0['method'] = 'anderson_rho'
summary_df_neb0 = pd.concat([summary_df_pearson_neb0, summary_df_anderson_neb0])
summary_df_means = pd.concat([summary_df_means, summary_df_neb0])

summary_df_means_pivot = summary_df_means.pivot(index=['n_extra_beh', 'true_rho', 'n_mri'], columns='method', values='rho_rmse').reset_index()
summary_df_means_pivot['rel_rmse'] = -((summary_df_means_pivot['corr_r'] - summary_df_means_pivot['anderson_rho']) / 
                                       (summary_df_means_pivot['corr_r'] - summary_df_means_pivot['corr_r1']))

summary_df_means_pivot['mean_rel_rmse'] = summary_df_means_pivot.groupby(['true_rho', 'n_mri'])['rel_rmse'].transform('mean')
summary_df_means_pivot['benefit'] = summary_df_means_pivot['mean_rel_rmse'] >= 0

summary_df_means_plot = summary_df_means.copy()
summary_df_means_plot['method'] = (pd.Categorical(summary_df_means_plot['method']).
                                     rename_categories({'corr_r': 'Pearson ($N_{nb}$)',
                                                        'corr_r1': 'Pearson ($N_{nb}$ + 1)',
                                                        'anderson_rho': 'Anderson'
                                                       }))

g = sns.relplot(x="n_extra_beh", y="rho_rmse", hue="method",
            col="true_rho", row="n_mri",
            kind="line", estimator=None, 
                hue_order=['Pearson ($N_{nb}$)', 'Pearson ($N_{nb}$ + 1)', 'Anderson'],
                facet_kws={'sharey': False},# 'margin_titles': True},
                data=summary_df_means_plot[(summary_df_means_plot['true_rho'] == .5) &
                                            (summary_df_means_plot['n_mri'] == 25) &
                                            summary_df_means_plot['method'].isin(
                                                ['Anderson', 'Pearson ($N_{nb}$)', 'Pearson ($N_{nb}$ + 1)'
                                                ])], aspect=1.5
               )

g.set_titles(r'$\rho$ = {col_name}, $N_{{nb}}$ = {row_name}')
g.set_xlabels(r'Behavioral Data ($N_{x}$)')
g.set_ylabels(r'$RMSE(\hat{\rho})$')
g.savefig(fig_dir + "fig_4.pdf", transparent=True, bbox_inches='tight')

sns.set_palette("deep", color_codes=True)
g = sns.relplot(x="n_extra_beh", y="rel_rmse", hue="benefit",
            row="true_rho", col="n_mri",
            kind="line", estimator=None, facet_kws={'sharey': False}, legend=False,
                palette=["b", "r"],
                data=summary_df_means_pivot
               )

g.set_titles(r'$\rho$ = {row_name} | $N_{{nb}}$ = {col_name}')
g.set_axis_labels("", "")
g.fig.supxlabel(r'Behavioral Data ($N_{x}$)')
g.fig.supylabel(r'$rel. \Delta RMSE(\hat{\rho})$')
g.refline(y=0)
g.refline(y=-1)

for axi in g.axes.flat:
    axi.yaxis.set_major_locator(matplotlib.pyplot.MaxNLocator(3))
g.savefig(fig_dir + "fig_5.pdf", transparent=True, bbox_inches='tight')

g = sns.relplot(x="n_extra_beh", y="rho_bias", hue="n_mri", 
            col="true_rho", legend='full', col_wrap = 5,
            kind="line", estimator=None, facet_kws={'sharey': False}, 
            data=summary_df_means_plot.query('method == "Anderson"'))
g.set_titles(r'$\rho$ = {col_name}')
g._legend.set_title('Neuroimaging Data ($N_{nb}$)')
sns.move_legend(g, "center left", bbox_to_anchor=(.92, 0.5))
g.set_axis_labels("", "")
g.fig.supxlabel(r'Behavioral Data ($N_{x}$)')
g.fig.supylabel(r'Bias')
g.savefig(fig_dir + "fig_6.pdf", transparent=True, bbox_inches='tight')

summary_df_means_supp_fig_2 = summary_df_means_all.query('method == ["anderson_rho", "anderson_rho_ddof0"]')

summary_df_means_supp_fig_2['method'] = (pd.Categorical(summary_df_means_supp_fig_2['method']).
                                     rename_categories({'anderson_rho_ddof0': 'Anderson (df = 0)',
                                                        'anderson_rho': 'Anderson (df = 1)',
                                                       }))

g = sns.relplot(x="n_extra_beh", y="rho_sq_error", hue="method",
            row="true_rho", col="n_mri",
                hue_order=['Anderson (df = 0)', 'Anderson (df = 1)'
                          ],
            kind="line", estimator=None, facet_kws={'sharey': False},
                data=summary_df_means_supp_fig_2
               )

g.set_titles(r'$\rho$ = {row_name} | $N_{{nb}}$ = {col_name}')
g.set_axis_labels("", "")
g.fig.supxlabel(r'Behavioral Data ($N_{x}$)')
g.fig.supylabel(r'Bias')

for axi in g.axes.flat:
    axi.yaxis.set_major_locator(matplotlib.pyplot.MaxNLocator(3))
g.savefig(fig_dir + "supp_fig_2.pdf", transparent=True, bbox_inches='tight')

g = sns.relplot(x="n_extra_beh", y="rho_rmse", hue="method",
            row="true_rho", col="n_mri",
                hue_order=['Pearson ($N_{nb}$)', 'Pearson ($N_{nb}$ + 1)',
                          'Anderson'
                          ],
            kind="line", estimator=None, facet_kws={'sharey': False},
                data=summary_df_means_plot
               )

g.set_titles(r'$\rho$ = {row_name} | $N_{{xy}}$ = {col_name}')

g.set_axis_labels("", "")
g.fig.supxlabel(r'Behavioral Data ($N_{x}$)')
g.fig.supylabel(r'$\Delta RMSE(\hat{\rho})$')

for axi in g.axes.flat:
    axi.yaxis.set_major_locator(matplotlib.pyplot.MaxNLocator(3))
g.savefig(fig_dir + "supp_fig_3.pdf", transparent=True, bbox_inches='tight')