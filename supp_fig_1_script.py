import numpy as np
import pandas as pd
import seaborn as sns
import getting_blood_from_a_stone_funcs as gbfs

sim_dir = 'sims/'

behav_dat = pd.read_csv("HCP_YA_subjects_2026_05_31_13_36_36.csv")

n_sims = 50000
n_mris = [50, 150, 250]
n_extra_behs = [0, 50, 100, 150, 200, 250, 300, 350, 400]

n_var = "FS_Total_GM_Vol"

b_vars = ["WM_Task_2bk_Acc", 
          "VSPLOT_TC", 
          "PMAT24_A_CR",
          "Language_Task_Acc",
          "CogTotalComp_AgeAdj",
          "Strength_AgeAdj", 
          "Emotion_Task_Median_RT"]

behav_dat_select = behav_dat[b_vars+[n_var]].dropna()

np.random.seed(0)
dfs = []
for b_var in b_vars:
    print(b_var)
    dats = [behav_dat_select.sample(n=650, replace=True) for i in range(n_sims)]
    true_rho = np.corrcoef(behav_dat_select[b_var], behav_dat_select[n_var])[0, 1]
    for n_mri in n_mris:
        print(n_mri)
        r = n_mri
        for n_extra_beh in n_extra_behs:
            print(n_extra_beh)
            n = n_mri + n_extra_beh
            dict_rho = {
                'sim_id': [i for i in range(n_sims)],
                'corr_r': [np.corrcoef(dats[i][b_var][:r], dats[i][n_var][:r])[0, 1] for i in range(n_sims)],
                'corr_r1': [np.corrcoef(dats[i][b_var][:(r + 1)], dats[i][n_var][:(r + 1)])[0, 1] for i in range(n_sims)],
                'anderson_rho': [gbfs.anderson_rho_ddof(dats[i][b_var][:n], dats[i][n_var][:r], ddof=1) for i in range(n_sims)],
            }
            df = pd.DataFrame(dict_rho)
            df['sim_id'] = np.arange(n_sims)
            df['n_mri'] = r
            df['n_extra_beh'] = n - r
            df['true_rho'] = true_rho
            df['b_var'] = b_var
            dfs.append(df)

summary_df = pd.concat(dfs, ignore_index=True)
summary_fp = 'supp_fig_1_summary_df.csv'
summary_df.to_csv(sim_dir+summary_fp, index=False)

value_vars = ['corr_r', 'corr_r1', 'anderson_rho']
summary_df_melt = pd.melt(summary_df, id_vars=['n_mri', 'n_extra_beh', 'b_var', 'true_rho', 'sim_id'], 
        value_vars=value_vars,
        var_name='method', value_name='rho')
summary_df_melt['rho_sq_error'] = (summary_df_melt['rho'] - summary_df_melt['true_rho']) ** 2
summary_df_melt['rho_bias'] = (summary_df_melt['rho'] - summary_df_melt['true_rho'])
summary_df_melt['rho_var'] = (summary_df_melt['rho'] - np.mean(summary_df_melt['rho'])) ** 2

summary_df_means = summary_df_melt.groupby(["n_mri", "method", "n_extra_beh", "true_rho", "b_var"]).mean().reset_index()
summary_df_means['rho_rmse'] = np.sqrt(summary_df_means['rho_sq_error'])

summary_fp = 'supp_fig_1_summary_df_means.csv'
summary_df_means.to_csv(sim_dir+summary_fp, index=False)

