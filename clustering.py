import numpy as np
import random
import statistics as stat

def K_Means(X, K):
    num_samps = X.shape[0]  # store number of samples
    
    ## Initialize random cluster centers
    cluster_index = random.sample(range(num_samps),K)
    newlabels = []
    clusters = []
    old_clusters = [0] * K
    
    for a in range(K):
        clusters.append(tuple(X[cluster_index[a]])) # random initial clusters
     
    for i in range(999): 
        if old_clusters == clusters:  # check if clusters stayed the same
            break
        else:
            old_clusters = clusters  # if not, old cluster is rewritten
            clusters = []  # clear clusters list
            newlabels = []  # clear newlabels list
            for b in range(num_samps):  # loop through samples
                samp = X[b]  # select sample
                centroid_dist = []
                
                for c in range(K):  # loop through centroids
                    centroid = old_clusters[c]  # initiate centroid as current cluster
                    
                    
                    samp_dist = np.square(samp-centroid)  # calculate sample distances
                    summed_dist = np.sum(samp_dist)  # sum of sample distances
                    centroid_dist.append(summed_dist)  # add to centroid distance list
                
                best_clust = centroid_dist.index(min(centroid_dist))    # find closest cluster for this sample              
                newlabels.append(best_clust)  # rewrite label for this sample
                
            for d in range(K):
                index_list = np.where(np.asarray(newlabels)==d)[0]  # find index of all samples closest to current centroid
                if len(index_list) == 0:  # if centroid is alone
                    clusters.append(old_clusters[d])  # make it the old one to avoid nan
                    clusters = sorted(clusters)  # sort cluster centers
                else:   
                    new_mean = np.nanmean(X[index_list,], axis=0)  # calculate new mean using closest samples
                    clusters.append(tuple(new_mean))  # add to clusters list
                    clusters = sorted(clusters)  # sort cluster centers
            
    C = clusters # C = clusters to be uniform to project
    C = np.array(C)  # converts C to numpy array
    return C

def K_Means_better(X, K):
    cluster_list = []  # initiate empty list of clusters
    
    for i in range(1000):  # iterate MANY times
        C = K_Means(X,K)  # store clusters of K_Means
        C = [tuple(s) for s in C]  # convert elements to tuple
        cluster_list.append(C)  # add to list of clusters
        
    final_list = [tuple(t) for t in cluster_list]  # convert list into tuple
    
    mode_cluster = stat.mode(final_list)  # find most common clusters from tuple
    mode_cluster = np.array(mode_cluster)   # converts tuple back to numpy array
    
    return mode_cluster