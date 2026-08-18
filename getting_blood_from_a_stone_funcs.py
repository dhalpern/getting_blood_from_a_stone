import numpy as np
import scipy as sp

# original anderson formulation is with ddof=0
def anderson_rho_ddof(x, y, ddof=0): 
    """
    corr missing data maximum likelihood,
    x is always the longer variable
    """
    
    n = len(x)
    r = len(y)
    x_r = x[:r]
    y_r = y[:r]
    
    rho_tilde = np.corrcoef(x_r, y_r)[0, 1]
    sigma_x_hat_sq = np.var(x, ddof=ddof)
    sigma_x_tilde_sq = np.var(x_r, ddof=ddof)
    sigma_x_ratio = sigma_x_hat_sq / sigma_x_tilde_sq
    rho_hat = rho_tilde * np.sqrt(sigma_x_ratio) * np.sqrt(1 / ((rho_tilde ** 2) * (sigma_x_ratio - 1) + 1))
    return rho_hat

def gen_data_md_rho(mu_x, mu_y, sigma_x, sigma_y, rho, N):
    cov = [[(sigma_x ** 2), (rho * sigma_x * sigma_y)],
             [(rho * sigma_x * sigma_y), (sigma_y ** 2)]]
    xy = np.random.multivariate_normal(size=N, mean=[mu_x, mu_y], cov=cov)
    x = xy[:, 0]
    y = xy[:, 1]

    data = {
        'B': x,
        'N': y
    }
    return data

def mu_hat(tau, y, sigma):
    mu_hat = sum(y / (sigma ** 2 + tau ** 2)) / sum(1 / (sigma ** 2 + tau ** 2))
    return mu_hat

def V_mu(tau, y, sigma):
    V_mu = 1 / sum(1 / (tau ** 2 + sigma ** 2))
    return V_mu
    
def sample_hier_model(y, sigma):
    n_grid = 2000
    n_sims = 1000
    J = len(y)
    
    tau_grid = np.linspace(.01, 40, num=n_grid)
    log_p_tau = np.zeros(n_grid)
    for i in range(n_grid):
        mu = mu_hat(tau_grid[i], y, sigma)
        V = V_mu(tau_grid[i], y, sigma)
        log_p_tau[i] = (.5 * np.log(V) - .5 * sum(np.log((sigma ** 2) + (tau_grid[i] ** 2))) - 
                        .5 * sum((y - mu) ** 2 / ((sigma ** 2) + (tau_grid[i] ** 2))))
    log_p_tau = log_p_tau - max(log_p_tau)
    p_tau = np.exp(log_p_tau)
    p_tau = p_tau / sum(p_tau)
    tau = np.random.choice(tau_grid, size=n_sims, p=p_tau)
    
    mu = np.zeros(n_sims)
    theta = np.zeros((n_sims, J))
    for i in range(n_sims):
        mu[i] = np.random.normal(size=1, loc=mu_hat(tau[i], y, sigma), scale=np.sqrt(V_mu(tau[i], y, sigma)))[0]
        theta_mean = (mu[i] / (tau[i] ** 2) + y / (sigma ** 2)) / (1 / (tau[i] ** 2) + 1 / (sigma ** 2))
        theta_sd = np.sqrt(1 / (1 / (tau[i] ** 2) + 1 / (sigma ** 2)))
        theta[i,] = np.random.normal(size=J, loc=theta_mean, scale=theta_sd)
    
    param_samps = {'mu': mu,
                   'tau': tau,
                   'theta': theta,
                   'theta_hat': np.mean(theta, axis=0),
                   'theta_hat_median': np.median(theta, axis=0)}
    return param_samps

def gen_data_hier(mu_x, sigma_x, low, high, N):
    tau = np.random.uniform(size=N, low=low, high=high)
    x = np.random.normal(size=N, loc=mu_x, scale=sigma_x)
    x_meas = np.random.normal(size=N, loc=x, scale=tau)
    data = {
        'theta': x,
        'B': x_meas,
        'sigma_B': tau,
    }
    return data

def group_t(rho, ICC, CNR, T, n):
    t = CNR * rho * np.sqrt((n * (T - 2)) / 
                      ((1 - ICC) + ((1 - rho ** 2) * (ICC + CNR ** 2)) + 
                        (ICC * (rho ** 2) * (T - 2))))
    return t

def t_test_power(t_stat, n, sig_level=.05, tside=2):
    nu = (n - 1)
    qu = sp.stats.t.isf((sig_level / tside), nu) 
    power = (sp.stats.t.sf(qu, nu, loc = np.sqrt(n) * t_stat) + 
             sp.stats.t.cdf(-qu, nu, loc = np.sqrt(n) * t_stat))
    return power

def solve_group_t(rho_diff, rho, ICC, CNR, T, n, n_extra):
    err = (t_test_power(group_t((rho + rho_diff), ICC, CNR, T, n), n) - 
           t_test_power(group_t(rho, ICC, CNR, T, (n + n_extra)), (n + n_extra)))
    return err