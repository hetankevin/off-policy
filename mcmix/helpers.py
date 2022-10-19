import numpy as np
from numba import jit, njit, prange
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy

def collect_sample(nsamples, mdp, pi_b, horizon, seed, iid=True):
    np.random.seed(seed)
    dataset = []
    for _ in range(nsamples):
        traj = mdp.generate_trajectory(pi_b, horizon, iid)
        dataset.append(traj)
    dataset = np.array(dataset)
    # x, a, u, x', r
    return dataset

def getSamplesMultiProc(samples, mdp, pi_b, horizon, iid=True):
    nprocs = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=nprocs, mp_context=multiprocessing.get_context('fork')) as executor:
        future = executor.map(collect_sample, [int(samples/nprocs) for i in range(nprocs)], repeat(copy.deepcopy(mdp)), 
                              repeat(copy.deepcopy(pi_b)), repeat(horizon), [i for i in range(nprocs)], repeat(iid))
    dataset = np.vstack(list(future))
    return dataset

# Gets \mathbb{P}_{\pi_b}(s' | s, a), 
#    the infinite-sample estimate of the transition probabilities
#    under the confounded behavior policy
# This is not the marginal transmission probability \mathbb{P}(s' | s, a),
#    as \mathbb{P}_{\pi_b}(s' | s, a) crucially weights the
#    behavior policy's tendency to take different actions
#    under different confounders
@njit(cache=True, parallel=False)
def getPb_spsa(nStates, nActions, u_dist, pi_b, pi_bsa, P):
    prob = np.zeros((nStates, nStates, nActions))
    for sp in range(nStates):
        for s in range(nStates):
            for a in range(nActions):
                for u in range(len(u_dist)):
                    prob[sp, s, a] += u_dist[u] * pi_b[u, s, a] * (1/pi_bsa[s,a]) * P[u, a, s, sp]
    return prob

# Gets counts of occupancies of state-action tuples in dataset
#    optional parameter burnin if one wants to only take counts past mixing time
def getN_sa(dataset, nStates, nActions, burnin=0):
    N_sa = np.zeros((nStates, nActions))
    for s,a,u,sp,r in dataset[:, burnin:, :].reshape(dataset[:, burnin:, :].shape[0]*dataset[:, burnin:, :].shape[1], 
                                                     dataset[:, burnin:, :].shape[2]):
        N_sa[int(s),int(a)] += 1
    return N_sa
