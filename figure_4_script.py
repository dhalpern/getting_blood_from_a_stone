import numpy as np
import pandas as pd
import os
import getting_blood_from_a_stone_funcs as gbfs

sim_dir = 'sims/'

mu_x = 0
mu_y = 0
sigma_x = 1
sigma_y = 1

max_sims = 1000000
n_sims = max_sims

# for testing
# n_sims = 1000

n_dats = n_sims
rhos = [.1, .3, .5, .7, .9]
n_extra_behs = [25, 50, 100, 250, 500, 1000]
n_mris = [25, 50, 100, 500, 1000]

for rho in rhos:
    print('rho', rho)
    print('generating data')
    np.random.seed(1)
    dats = [gbfs.gen_data_md_rho(mu_x, mu_y, sigma_x, sigma_y, rho, 2000) for i in range(n_dats)]
    print('simulating fits')
    for n_mri in n_mris:
        print('n_mri', n_mri)
        summary_fp = 'sims/rho'+str(rho)+'_n_mri'+str(n_mri)+'_n_sims'+str(n_sims)+'_fig4_summary_df_means.csv'
        if not os.path.exists(summary_fp):
            dfs = []
            r = n_mri
            for n_extra_beh in n_extra_behs:
                # need more precision for lower correlations
                print('n_extra_beh', n_extra_beh)
                n = n_mri + n_extra_beh
                dict_rho = {
                    'sim_id': [i for i in range(n_sims)],
                    'corr_r': [np.corrcoef(dats[i]['B'][:r], dats[i]['N'][:r])[0, 1] for i in range(n_sims)],
                    'corr_r1': [np.corrcoef(dats[i]['B'][:(r + 1)], dats[i]['N'][:(r + 1)])[0, 1] for i in range(n_sims)],
                    'anderson_rho': [gbfs.anderson_rho_ddof(dats[i]['B'][:n], dats[i]['N'][:r], ddof=1) for i in range(n_sims)],
                    'anderson_rho_ddof0': [gbfs.anderson_rho_ddof(dats[i]['B'][:n], dats[i]['N'][:r], ddof=0) for i in range(n_sims)],
                }
                df = pd.DataFrame(dict_rho)
                df['sim_id'] = np.arange(n_sims)
                df['n_mri'] = r
                df['n_extra_beh'] = n - r
                df['true_rho'] = rho
                dfs.append(df)

            summary_df = pd.concat(dfs, ignore_index=True)

            value_vars = ['corr_r', 'corr_r1', 'anderson_rho', 'anderson_rho_ddof0']
            summary_df_melt = pd.melt(summary_df, id_vars=['n_mri', 'n_extra_beh', 'true_rho'], 
                    value_vars=value_vars,
                    var_name='method', value_name='rho')
            summary_df_melt['rho_sq_error'] = (summary_df_melt['rho'] - summary_df_melt['true_rho']) ** 2
            summary_df_melt['rho_bias'] = (summary_df_melt['rho'] - summary_df_melt['true_rho'])
            summary_df_melt['rho_var'] = (summary_df_melt['rho'] - np.mean(summary_df_melt['rho'])) ** 2

            summary_df_means = summary_df_melt.groupby(["n_mri", "method", "n_extra_beh", "true_rho"]).mean().reset_index()

            summary_df_means['rho_rmse'] = np.sqrt(summary_df_means['rho_sq_error'])

            summary_df_means.to_csv(summary_fp, index=False)
            if n_sims == max_sims:
                summary_df_melt.to_csv('sims/rho'+str(rho)+'_n_mri'+str(n_mri)+'_fig4_summary_df_melt.csv', index=False)
        else:
            print('file already exists')
    del dats, dfs