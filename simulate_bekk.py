import numpy as np
import bekk
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import multiprocessing as mp

from datetime import datetime
import os


def calculate_portfolio_var(w,V):
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def gmv(V):
  n = V.shape[0]
  w0 = np.repeat(1/n, n)
  cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1})
  res = minimize(calculate_portfolio_var, w0, args=V, method='SLSQP', constraints=cons)
  return res.x


def generate_bekk(p, T, params, type):
    """
    Generate Diagonal Multivariate BEKK(1,1) Garch Series
    Input:
    p         Dimension of series (number of stocks)
    T         Length of series (number of observations)
    params    Array containing diagonal a and b parameters
    type      'Scalar', 'Diagonal' or 'Full'  

    Output:
    e         (T x K) data matrix
    H         H_1, ... , H_T conditional covariance (K x K x T) matrix
    w         True Global Minimum Variance (GMV) portfolio weights at time T
    data      Data wrapped as a bekk class
    a         True scalar parameter a
    b         True scalar parameter b
    C         True matrix C
    params      BEKKParams instance
    """

    if type == 'diagonal':
        A = np.array(((params[0], 0), (0, params[1])))
        B = np.array(((params[2], 0), (0, params[3])))
    elif type == 'scalar':
        A = np.eye(p) * params[0]
        B = np.eye(p) * params[1]
    elif type == 'full':
        A = np.array(((params[0], params[1]), (params[2], params[3])))
        B = np.array(((params[4], params[5]), (params[6], params[7])))

    target = np.eye(p)
    params = bekk.ParamStandard.from_target(amat=A, bmat=B, target=target)
    e, H = bekk.generate_data.simulate_bekk(params, nobs=T, distr='normal')
    data = bekk.BEKK(e)

    a = params.amat[0,0]
    b = params.bmat[0,0]
    C = params.cmat

    w = gmv(H[T-1])

    return e, H, w, data, a, b, C, params


def bootstrap(b, intercept, hvar, z_hat, T, p):
    """
    Calculate GMV weights for bootstrap sample b
    Input:
    b                 Bootstrap sample number (b in range(0,B))    
    intercept         C'C 
    hvar              H covariance matrices
    z_hat             Calculated error terms
    T
    p
    
    Output:
    GMV weights for bootstrap sample b
    """
    hboot = np.zeros((T, p, p))
    eboot = np.zeros((T, p))

    for t in range(0, T):
      if t == 0:
        hboot[t] = hvar[t]
      else:
        eboot2 = eboot[t-1, np.newaxis].T * eboot[t-1]
        hboot[t] = intercept + bekk.fit.param_final.amat.dot(eboot2).dot(bekk.fit.param_final.amat.T) + bekk.fit.param_final.bmat.dot(hvar[t-1]).dot(bekk.fit.param_final.bmat.T)
      zboot = z_hat[np.random.choice(z_hat.shape[0]),:]
      hboot12 = np.linalg.cholesky(hboot[t])
      eboot[t] = hboot12.dot(np.atleast_2d(zboot).T).flatten()
    return gmv(hboot[T-1])


def bootstrap_CI(data, B, params, bekk_type='diagonal'):
    """
    Calculate bootstrap confidence intervals for Scalar BEKK(1,1)
    Input:
    data    Data wrapped as bekk class
    B       Number of bootstrap replicates (i.e. 999, 1499)
    params    BEKKParams instance
    bekk_type      'scalar', 'diagonal' or 'full'

    Output:
    intervals   Confidence intervals for each parameter
    """

    T = data.innov.shape[0]
    p = data.innov.shape[1]
    e = data.innov

    # 1. Estimate (a, b, C) using QML
    bekk.fit = data.estimate(param_start=params, restriction=bekk_type, model='standard', method='SLSQP')

    # 2. Estimate the conditional var/covariances
    hvar = np.zeros((T, p, p))
    hvar[0] = bekk.fit.param_final.get_uvar()
    intercept = bekk.fit.param_final.cmat.dot(bekk.fit.param_final.cmat.T)

    for t in range(1, T):
      e2 = e[t-1, np.newaxis].T * e[t-1]
      hvar[t] = intercept + bekk.fit.param_final.amat.dot(e2).dot(bekk.fit.param_final.amat.T) + bekk.fit.param_final.bmat.dot(hvar[t-1]).dot(bekk.fit.param_final.bmat.T)

    # 3. Compute residuals by the standardized observations
    z_hat = np.zeros((T, p))
    for t in range(0,T):
      hvarinv12 = np.linalg.inv(np.linalg.cholesky(hvar[t]))
      z_hat[t] = hvarinv12.dot(np.atleast_2d(e[t]).T).flatten()

    # 4. Replicate for B bootstrap samples
    w_boots = np.zeros((B, p))
    arg_iterable = [(b, intercept, hvar, z_hat, T, p) for b in range(0,B)]

    with mp.Pool(os.cpu_count()-1) as pool:
      results = pool.starmap(bootstrap, arg_iterable)
    
    results = np.array(results)
    results.reshape((B,p))

    # 6. Compute confidence intervals
    w_intervals = np.zeros((p, 2))
    for i in range(0, p):
      w_intervals[i,:] = np.quantile(results[:,i], (0.025, 0.975))
    print(w_intervals)

    return w_intervals, bekk.fit, w_boots


def run_simulation(S, B, p, T, params, bekk_type):
    """
    Run simulations of Scalar BEKK(1,1) to calculate coverage probablity of GMV weights

    Input:
    S         Number of simulations
    B         Number of bootstrap replicates
    p         Number of series to simulate
    T         Number of observations to simulate
    params      Scalar BEKK(1,1) parameters
    bekk_type        'Scalar', 'Diagonal' or 'Full'
    """
    coverage = np.array(np.repeat(0,p))
    f = open("dbekk_{}.txt".format(T), "a")
    f.write("STARTING {} SIMULATIONS FOR T={}, B={}, p={} \n".format(S, T, B, p))
    f.write("-------------------------------------------- \n")
    f.close()

    for s in range(0,S):
      start=datetime.now()
      e_sim, H_sim, w_sim, data_sim, a_sim, b_sim, C_sim, params_sim = generate_bekk(p=p, T=T, params=params, bekk_type=bekk_type)
      w_intervals, bekk_fit, w_boot = bootstrap_CI(data=data_sim, B=B, params=params_sim)
      for i in range(0, p):
        if w_intervals[i,0] <= w_sim[i] and w_intervals[i,1] >= w_sim[i]:
          coverage[i] = coverage[i] + 1
      total_time = datetime.now()-start
      f = open("dbekk_{}.txt".format(T), "a")
      f.write("SIMULATION {} COMPLETED -- Took {} seconds.. \n".format(s, total_time))
      f.close()
    cp = coverage/S
    print(cp)
    f = open("dbekk_{}.txt".format(T), "a")
    f.write("{} \n".format(cp))
    f.close()
    return cp


if __name__ == '__main__':
    run_simulation(S=10, B=1499, p=2, T=500, params=[0.5, 0.3, 0.4, 0.8], bekk_type='diagonal')
    run_simulation(S=10, B=1499, p=2, T=1000, params=[0.5, 0.3, 0.4, 0.8])
    run_simulation(S=10, B=1499, p=2, T=2000, params=[0.5, 0.3, 0.4, 0.8])
    run_simulation(S=10, B=1499, p=2, T=3000, params=[0.5, 0.3, 0.4, 0.8])





