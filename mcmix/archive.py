
# NUMBA STAT CALCULATION
'''

@njit(parallel=True, fastmath=True)
def getSims(onehotsaclust, hs, Vsa):
    M = onehotsaclust.shape[0]
    S = onehotsaclust.shape[2]
    A = onehotsaclust.shape[3]
    statmns = np.zeros((M,M))
    for m in prange(M):
        if m % 100 == 0:
            print(m)
        for n in range(M):
            if m < n:
                break
            statg = np.zeros(G)
            for g in range(G):
                maxstat = 0
                for s in range(S):
                    for a in range(A):
                        stat = ((hs[g,0,m,s,a] - hs[g,0,n,s,a]).T @ Vsa[s,a] 
                                 @ Vsa[s,a].T @ (hs[g,1,m,s,a] - hs[g,1,n,s,a]))
                        if stat > maxstat:
                            maxstat = stat
                statg[g] = maxstat
            statmns[m,n] = np.median(statg)
    return statmns + statmns.T - np.diag(np.diag(statmns))

statmns = getSims(onehotsaclust, hs, eigvecsa)


@njit(parallel=True, fastmath=True)
def getSims(onehotsaclust, h1, h2, Vsa):
    M = onehotsaclust.shape[0]
    S = onehotsaclust.shape[2]
    A = onehotsaclust.shape[3]
    statmns = np.zeros((M,M))
    for m in prange(M):
        if m % 100 == 0:
            print(m)
        for n in range(M):
            if m < n:
                break
            maxstat = 0
            for s in range(S):
                for a in range(A):
                    stat = ((h1[m,s,a] - h1[n,s,a]).T @ Vsa[s,a] 
                             @ Vsa[s,a].T @ (h2[m,s,a] - h2[n,s,a]))
                    if stat > maxstat:
                        maxstat = stat
            statmns[m,n] = maxstat
    return statmns + statmns.T - np.diag(np.diag(statmns))

statmns = getSims(onehotsaclust, 
                  geth(onehotsaclust[:,omegaone,:,:], onehotspclust[:,omegaone,:]),
                  geth(onehotsaclust[:,omegatwo,:,:], onehotspclust[:,omegatwo,:]), 
                  eigvecsa)
                  
                  
                  
@njit(parallel=True, fastmath=True)
def getSims(onehotsaclust, proj1, proj2, Vsa):
    M = onehotsaclust.shape[0]
    S = onehotsaclust.shape[2]
    A = onehotsaclust.shape[3]
    statmns = np.zeros((M,M))
    for m in prange(M):
        if m % 100 == 0:
            print(m)
        for n in range(M):
            if m < n:
                break
            maxstat = 0
            for s in range(S):
                for a in range(A):
                    stat = ((proj1[m,s,a] - proj1[n,s,a]) @ (proj2[m,s,a] - proj2[n,s,a]))
                    if stat > maxstat:
                        maxstat = stat
            statmns[m,n] = maxstat
    return statmns + statmns.T - np.diag(np.diag(statmns))

statmns2 = getSims(onehotsaclust, 
                  proj1, proj2, 
                  eigvecsa)


proj1 = (geth(onehotsaclust[:,omegaone,:,:], onehotspclust[:,omegaone,:])[..., None,:] @ eigvecsa[None,...]).squeeze()
proj2 = (geth(onehotsaclust[:,omegatwo,:,:], onehotspclust[:,omegatwo,:])[..., None,:] @ eigvecsa[None,...]).squeeze()
statmns = ((proj1[None,...] - proj1[:,None,...]) * (proj2[None,...] - proj2[:,None,...])).sum(-1).max(axis=(2,3))

projs = (hs[..., None,:] @ eigvecsa[None,...]).squeeze()
statmns2 = np.median(((projs[:,0,None,...] - projs[:,0,:,None,...]) * 
            (projs[:,1,None,...] - projs[:,1,:,None,...])).sum(-1).max(axis=(3,4)), axis=0)
            
d_sa = (N_sa/N_sa.sum())


'''



#PROJECTION NUMPY VECTORIZATION
#Deprecated in favor of tensorflow

'''
proj1 = (geth(onehotsaclust[:,omegaone,:,:], onehotspclust[:,omegaone,:])[..., None,:] @ eigvecsa[None,...]).squeeze()
proj2 = (geth(onehotsaclust[:,omegatwo,:,:], onehotspclust[:,omegatwo,:])[..., None,:] @ eigvecsa[None,...]).squeeze()
statmns = ((proj1[None,...] - proj1[:,None,...]) * (proj2[None,...] - proj2[:,None,...])).sum(-1).max(axis=(2,3))

projs = (hs[..., None,:] @ eigvecsa[None,...]).squeeze()
statmns2 = np.median(((projs[:,0,None,...] - projs[:,0,:,None,...]) * 
            (projs[:,1,None,...] - projs[:,1,:,None,...])).sum(-1).max(axis=(3,4)), axis=0)
            
d_sa = (N_sa/N_sa.sum())
'''


#MEDIAN OF MEANS ESTIMATOR
#Deprecated, no theoretical value

'''
d_sat = tf.convert_to_tensor(N_sa/N_sa.sum(), np.float64)
statmns2 = tfp.stats.percentile(
                tf.reduce_max(
                    tf.reduce_sum(
                        (projst[:,0,None,...] - projst[:,0,:,None,...]) * 
                        (projst[:,1,None,...] - projst[:,1,:,None,...]), 
                    axis=-1), 
                axis=(3,4)), 
            axis=0, q=50.0, interpolation='midpoint').numpy()
'''

#PYTORCH MEDIAN OF MEANS CODE
# deprecated, crashes instantly due to 500GB ram demand
'''
import torch
device = torch.device("mps")
import seaborn as sns
plt.figure(figsize=(16,9))
plt.hist(statmns2.flatten(), bins=100, density=True)[2]
plt.hist(statmns2.flatten(), bins=onehotsaclust.shape[0], density=True)[2]
sns.kdeplot(statmns2.flatten(), bw_adjust=0.5)
hsp = torch.tensor(hs.astype(np.float32), device=device)
eigvecsap = torch.tensor(eigvecsa.astype(np.float32), device=device)
h1p = torch.tensor(geth(onehotsaclust[:,omegaone,:,:], onehotspclust[:,omegaone,:]).astype(np.float32), device=device)
h2p = torch.tensor(geth(onehotsaclust[:,omegatwo,:,:], onehotspclust[:,omegatwo,:]).astype(np.float32), device=device)
d_sap = torch.tensor((N_sa/N_sa.sum()).astype(np.float32), device=device)
proj1p = torch.squeeze(h1p[..., None,:] @ eigvecsap[None,...])
proj2p = torch.squeeze(h2p[..., None,:] @ eigvecsap[None,...])
statmnsp = ((proj1p[None,...] - proj1p[:,None,...]) * (proj2p[None,...] - proj2p[:,None,...])).sum(-1).amax(axis=(2,3))
projsp = torch.squeeze(hsp[..., None,:] @ eigvecsap[None,...])
statmns2p = torch.median(((projsp[:,0,None,...] - projsp[:,0,:,None,...]) * 
            (projsp[:,1,None,...] - projsp[:,1,:,None,...])).sum(-1).max(axis=(3,4)), axis=0)
'''



## weighted sum similarity code
#deprecated, poor performance

'''
temp = tf.reduce_sum(
        (projst[:,0,None,...] - projst[:,0,:,None,...]) * 
        (projst[:,1,None,...] - projst[:,1,:,None,...]), 
        axis=-1)

statmns3 = tfp.stats.percentile(
            tf.reduce_sum(temp * d_sat[None, None, None, ...], axis=(3,4)),
        axis=0, q=50.0, interpolation='midpoint').numpy()

statmns4 = tf.reduce_sum(
                    tf.reduce_sum(
                        (proj1t[None,...] - proj1t[:,None,...]) * 
                        (proj2t[None,...] - proj2t[:,None,...]), 
                    axis=-1) * d_sat[None, None, ...],
                axis=(2,3)).numpy()
'''

