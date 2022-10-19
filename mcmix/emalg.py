import numpy as np
from numba import jit, njit, prange
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import helpers

@njit(parallel=False, fastmath=True)
def getmodelestim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions):
    Phat_ksa = np.zeros((K, nStates, nActions, nStates))
    for i in prange(states.shape[0]):
        for j in range(states.shape[1]):
            Phat_ksa[int(clusterlabs[i]), 
                     int(states[i,j]), 
                     int(actions[i,j]), 
                     int(nextstates[i,j])] += 1
    return Phat_ksa

@njit(parallel=False, fastmath=True)
def getmodelestimsoft(expect, states, actions, nextstates,
                  K, nStates, nActions):
    Phat_ksa = np.zeros((K, nStates, nActions, nStates))
    for i in prange(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(K):
                Phat_ksa[k, 
                         int(states[i,j]), 
                         int(actions[i,j]), 
                         int(nextstates[i,j])] += expect[k,i]
    return Phat_ksa

def getModelEstim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions, hard=True):
    if hard:
        model = getmodelestim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions)
    else:
        model = getmodelestimsoft(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions)
    return model / np.nansum(model, axis=-1)[..., None]


def getloglik(expect, modelestim, states, actions, nextstates, hard=True):
    if hard:
        return np.nansum(
                    np.nansum(
                        np.log(modelestim[:, states, actions, nextstates]), 
                    axis=-1)[expect, np.arange(len(states))])
    else:
        return np.nansum(
                    np.nansum(
                        np.log(np.nansum(modelestim[:, states, actions, nextstates] 
                                        * expect[...,None], axis=0)),
                    axis=-1))

def classify(model, states, actions, nextstates, reg=0, prior=None, startweights=None, labs=True):
    if startweights is not None:
        probs = np.nansum(np.log(model[:, states, actions, nextstates]) 
                          + np.log(startweights[:,states]), axis=-1)
    else:
        probs = np.nansum(np.log(model[:, states, actions, nextstates]), axis=-1)
    probs += np.random.uniform(high=1e-7, size=probs.shape)
    if reg > 0:
        probs += reg*np.log(prior)[:,None]
    if labs:
        return probs.argmax(0)
    else:
        return probs
    
def em(expect, modelestim, states, actions, nextstates, labels, 
                K, nStates, nActions, prior, 
               max_iter = 100, checkin=5, reg = 0,
               permute=False, permutation=0, verbose=True, hard=True):
    i = 0
    modelold = np.ones(modelestim.shape)
    while i == 0 or np.nansum(np.abs(modelold - modelestim)) > 1e-3:
        modelold = modelestim
        
        if hard:
            modelestim = getModelEstim(expect.astype(int), states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=True)
            startweights = helpers.getStartWeights(states, expect, K, nStates, hard=True)
            expectprobs = np.nansum(np.log(modelestim[:, states, actions, nextstates])
                                    + np.log(startweights[:,states]), -1)
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expect = (expectprobs + #random number to perturb argmax 
                      reg*np.log(prior)[:,None]).argmax(0)
            prior = np.bincount(expect)/len(expect)
        else:
            modelestim = getModelEstim(expect, states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=False)
            startweights = helpers.getStartWeights(states, expect, K, nStates, hard=False)
            expectprobs = np.nansum(np.log(modelestim[:, states, actions, nextstates]) 
                                    + np.log(startweights[:,states]), axis=-1)
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expect = np.exp(expectprobs + 
                            reg*np.log(prior)[:,None])
            expect = (expect / np.nansum(np.abs(expect), axis=0))
            prior = np.bincount(expect.argmax(0))/len(expect)
        
        i += 1
        if i % checkin == 0 and verbose:
            print('iteration', i, 'diff', np.nansum(np.abs(modelold - modelestim)))
            expectlabs = expect if hard else expect.argmax(0)
            if permute:
                print('accuracy:', max(np.mean(expectlabs == labels), np.mean(expectlabs != labels)))
                    
            else:
                print('accuracy:', [np.mean(expectlabs == labels), np.mean(expectlabs != labels)][permutation])
            print(getloglik(expect, modelestim, states, actions, nextstates, hard=hard))
        if i > max_iter:
            break
    loglik = getloglik(expect, modelestim, states, actions, nextstates, hard=hard)
    if verbose:
        print('log-likelihood:', loglik)
    return expect, modelestim, loglik