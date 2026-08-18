import numpy as np
import pandas as pd
import getting_blood_from_a_stone_funcs as gbfs

sim_dir = 'sims/'

n_sims = 10000
dats = {}
log_sigma_diff = -1
log_sigma_M_diff = -1
log_sigma_x = 0

lsd = log_sigma_diff
lsMd = log_sigma_M_diff
log_sigma_M = log_sigma_x + lsd
log_low = log_sigma_M - lsMd
log_high = log_sigma_M + lsMd
low = np.exp(log_low)
high = np.exp(log_high)
sigma_x = np.exp(log_sigma_x)
print(sigma_x, low, high)
dats = [gbfs.gen_data_hier(0, sigma_x, low, high, 500) for i in range(n_sims)]

dfs = []
n_extra_behs = [0, 25, 50, 100, 200]
n_mris = [25, 50, 100]
for n_mri in n_mris:
    print(n_mri)
    r = n_mri
    for n_extra_beh in n_extra_behs:
        print(n_extra_beh)
        n = n_mri + n_extra_beh
        unif_corr_thetas = np.zeros(n_sims)       
        
        for i in range(n_sims):
            unif_hier_samps = gbfs.sample_hier_model(dats[i]['B'][:n], dats[i]['sigma_B'][:n])
            unif_corr_thetas[i] = np.corrcoef(dats[i]['theta'][:r], unif_hier_samps['theta_hat'][:r])[0, 1]
        dict_rho = {
            'sim_id': [i for i in range(n_sims)],
            'unif_corr_thetas': unif_corr_thetas,
        }
        
        df = pd.DataFrame(dict_rho)
        df['sim_id'] = np.arange(n_sims)
        df['n_mri'] = r
        df['n_extra_beh'] = n - r
        dfs.append(df)

summary_df = pd.concat(dfs, ignore_index=True)

summary_df_corr_melt = pd.melt(summary_df, id_vars=['n_mri', 'n_extra_beh'], 
        value_vars='unif_corr_thetas',
        var_name='method', value_name='corr')

summary_df_corr_means = summary_df_corr_melt.groupby(['n_mri', 'n_extra_beh', 'method']).mean().reset_index()

summary_df_corr_means.to_csv(sim_dir+'fig_7_summary_df_corr_means.csv', index=False)