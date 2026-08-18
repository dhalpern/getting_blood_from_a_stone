import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=False)
sns.set_style("white")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3})
fig_dir = 'figures/'

sim_dir = 'sims/'

data_dict = pd.read_csv("HCP_S1200_DataDictionary_Oct_30_2023.csv")
hcp_dat = pd.read_csv("HCP_YA_subjects_2026_05_31_13_36_36.csv")

n_var = ["FS_Total_GM_Vol"]
b_vars = ["WM_Task_2bk_Acc", 
          "VSPLOT_TC", 
          "PMAT24_A_CR",
          "Language_Task_Acc",
          "CogTotalComp_AgeAdj",
          "Strength_AgeAdj", 
          "Emotion_Task_Median_RT"]
behav_corr_mat = hcp_dat[n_var+b_vars].corr()
behav_corr_df = behav_corr_mat['FS_Total_GM_Vol'].reset_index(name='corr')
behav_corr_label_df = behav_corr_df.merge(data_dict, left_on='index', right_on='columnHeader')
behav_corr_label_df['label'] = behav_corr_label_df['assessment'] + ' \n($r = $' + behav_corr_label_df['corr'].round(1).astype(str) + ')'
behav_corr_label_df['short_label'] = behav_corr_label_df['columnHeader'] + ' \n($r = $' + behav_corr_label_df['corr'].round(1).astype(str) + ')'

summary_df_means = pd.read_csv(sim_dir+'supp_fig_1_summary_df_means.csv')

summary_df_means_pivot = summary_df_means.pivot(index=['n_extra_beh', 'true_rho', 'n_mri', 'b_var'], columns='method', values='rho_rmse').reset_index()
summary_df_means_pivot['rel_rmse'] = -((summary_df_means_pivot['corr_r'] - summary_df_means_pivot['anderson_rho']) / 
                                       (summary_df_means_pivot['corr_r'] - summary_df_means_pivot['corr_r1']))

plot_df = summary_df_means_pivot.merge(behav_corr_label_df, left_on='b_var', right_on='columnHeader')

g = sns.relplot(x="n_extra_beh", y="rel_rmse", 
                col='short_label',
                row="n_mri",
            kind="line", estimator=None, 
                facet_kws={
                    'sharey': False, 'margin_titles': True},
                data=plot_df
               )

g.set_titles(col_template="{col_name}", row_template="{row_name} MRI Subjects")
g.set_axis_labels("", "")
g.fig.supxlabel(r'Behavioral Data ($N_{x})$')
g.fig.supylabel(r'$rel. \Delta RMSE(\hat{\rho})$')
g.refline(y=0)
g.refline(y=-1)

for axi in g.axes.flat:
    axi.yaxis.set_major_locator(matplotlib.pyplot.MaxNLocator(3))
g.savefig(fig_dir+"supp_fig_1.pdf", transparent=True, bbox_inches='tight')
