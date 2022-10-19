import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')

## ALGORITHM: CLUSTERING

## TEST STATISTIC CALCULATION
'''
def computeStat(h1, h2, eigvecsa, device='/CPU:0'):
    with tf.device(device):
        h1t = tf.convert_to_tensor(h1, np.float64)
        h2t = tf.convert_to_tensor(h2, np.float64)
        eigvecsat = tf.convert_to_tensor(eigvecsa, np.float64)
        proj1t = tf.squeeze(h1t[..., None,:] @ eigvecsat[None,...])
        proj2t = tf.squeeze(h2t[..., None,:] @ eigvecsat[None,...])
        statmns = tf.reduce_max(
                        tf.reduce_sum(
                            (proj1t[None,...] - proj1t[:,None,...]) * 
                            (proj2t[None,...] - proj2t[:,None,...]), 
                        axis=-1),
                    axis=(2,3)).numpy()
    return statmns
'''

def computeStat(hs, eigvecsa, device='/CPU:0', proj=True):
    with tf.device(device):
        hst = tf.convert_to_tensor(hs, np.float64)
        eigvecsat = tf.convert_to_tensor(eigvecsa, np.float64)
        if proj:
            projst = tf.squeeze(hst[..., None,:] @ eigvecsat[None,...])
        else:
            projst = hst
        statmns2 = tf.reduce_max(
                            tf.reduce_sum(
                                (projst[0,None,...] - projst[0,:,None,...]) * 
                                (projst[1,None,...] - projst[1,:,None,...]), 
                            axis=-1), 
                    axis=(2,3)).numpy()
    return statmns2

## OBTAINING CLUSTERS
def getClusters(statmns, thresh, K, method='kmeans'):
    return sklearn.cluster.spectral_clustering((statmns < thresh).astype(int), n_clusters=K,
                                                         assign_labels='kmeans')

## DIAGNOSTICS
def clusterDiagnostics(statmns, K, labels, lo, hi, step, method='kmeans'):
    accs = []
    wts = []
    taus = np.arange(lo, hi, step)
    for tau in tqdm(taus):
        clusterlabs = getClusters(statmns, thresh=tau, K=K, method=method)
        accs.append(max(np.mean(clusterlabs == labels), 
                        np.mean(clusterlabs != labels)))
        wts.append(max(np.mean(clusterlabs==1), np.mean(clusterlabs==0)))
    plt.figure(figsize=(16,9))
    plt.plot(taus, 100*np.array(accs), label='Accuracies (%)')
    plt.plot(taus, 100*np.array(wts), label='Max. Cluster Weight (%)')
    plt.xlabel('Threshold')
    plt.title('Accuracy against thresholds')
    plt.legend()
    