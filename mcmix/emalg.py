import numpy as np
from numba import jit, njit, prange
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import helpers


# Gets starting state probabilities per confounder
def getStartWeights(states, predlabs, K, nStates, hard=True):
    wts = np.zeros((K, nStates))
    if hard:
        for i in range(len(states)):
            wts[predlabs[i],states[i, 0]] += 1
        return wts/np.array([sum(predlabs==k) for k in range(K)])[:,None]
    else:
        for i in range(len(states)):
            for k in range(K):
                wts[k,states[i, 0]] += predlabs[k, i]
        return wts/np.sum(wts, axis=1)[:,None]

@njit(parallel=False, cache=True)
def getPolicyHelperSoft(states, actions, K, nStates, nActions, predprobs):
    pi_ksa = np.zeros((K, nStates, nActions)) #k, s, a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(K):
                pi_ksa[int(k), 
                       int(states[i,j]), 
                       int(actions[i,j])] += predprobs[k, i]
    return pi_ksa

@njit(parallel=False, cache=True)
def getPolicyHelperLabs(states, actions, K, nStates, nActions, predlabs):
    pi_ksa = np.zeros((K, nStates, nActions)) #k, s, a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            pi_ksa[int(predlabs[i]), 
                   int(states[i,j]), 
                   int(actions[i,j])] += 1
    return pi_ksa

@njit(parallel=False, cache=True)
def getPolicyHelper(states, actions, nStates, nActions):
    pi_sa = np.zeros((nStates, nActions)) #s,a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            pi_sa[int(states[i,j]), int(actions[i,j])] += 1
    return pi_sa

def getPolicyEstim(states, actions, K, nStates, nActions, preds=None, hard=True):
    if not hard:
        pi = getPolicyHelperSoft(states, actions, K, nStates, nActions, preds)
    elif preds is not None:
        pi = getPolicyHelperLabs(states, actions, K, nStates, nActions, preds)
    else:
        pi = getPolicyHelper(states, actions, nStates, nActions)
    pi = pi/np.nansum(pi, axis=-1)[...,None]
    pi[np.isnan(pi)] = 1/nActions
    return pi

@njit(parallel=False, cache=True)
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

@njit(parallel=False, cache=True)
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
    model = model / np.nansum(model, axis=-1)[..., None]
    model[np.isnan(model)] = 1/nStates
    return model


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

def classify(model, states, actions, nextstates, policy=None, reg=0, prior=None, startweights=None, labs=True):
    if policy is not None:
        probs = np.nansum(np.log(model[:, states, actions, nextstates])
                    + np.log(policy[:, states, actions]), axis=-1)
    else:
        probs = np.nansum(np.log(model[:, states, actions, nextstates]), axis=-1)
    if startweights is not None:
        probs += np.log(startweights[:,states[:,0]])
    
    probs += np.random.uniform(high=1e-7, size=probs.shape)
    if reg > 0:
        probs += reg*np.log(prior)[:,None]
    if labs:
        return probs.argmax(0)
    else:
        return probs
    
def em(expect, modelestim, states, actions, nextstates, labels, 
                K, nStates, nActions, prior, 
               max_iter = 100, min_iter = 10, checkin=5, reg = 0,
               permute=False, permutation=0, verbose=True, hard=True):
    i = 0
    modelold = np.ones(modelestim.shape)
    while i < min_iter or np.nansum(np.abs(modelold - modelestim)) > 1e-3:
        modelold = modelestim
        
        if hard:
            policy = getPolicyEstim(states, actions, 
                         K, nStates, nActions, expect, hard=True)
            modelestim = getModelEstim(expect.astype(int), states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=True)
            startweights = getStartWeights(states, expect, K, nStates, hard=True)
            expectprobs = np.nansum(np.log(modelestim[:, states, actions, nextstates])
                                    + np.log(policy[:, states, actions]), -1) + np.log(startweights[:,states[:,0]])
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expect = (expectprobs + #random number to perturb argmax 
                      reg*np.log(prior)[:,None]).argmax(0)
            prior = np.bincount(expect)/len(expect)
        else:
            policy = getPolicyEstim(states, actions, 
                         K, nStates, nActions, expect, hard=False)
            modelestim = getModelEstim(expect, states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=False)
            startweights = getStartWeights(states, expect, K, nStates, hard=False)
            expectprobs = (np.nansum(np.log(modelestim[:, states, actions, nextstates]) 
                                    + np.log(policy[:, states, actions]), axis=-1) 
                           + np.log(startweights[:,states[:,0]]))
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expect = np.exp(expectprobs + 
                            reg*np.log(prior)[:,None])
            expect = (expect / np.nansum(np.abs(expect), axis=0))
            prior = np.bincount(expect.argmax(0))/len(expect) #fix to soft prior
        
        i += 1
        if i % checkin == 0 and verbose:
            print('iteration', i, 'diff', np.nansum(np.abs(modelold - modelestim)))
            expectlabs = expect if hard else expect.argmax(0)
            print(len(expectlabs), len(labels))
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