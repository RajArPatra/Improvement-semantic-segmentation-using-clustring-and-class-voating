# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:37:15 2020

@author: kb
"""
import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics.pairwise import pairwise_distances


class Quickshift():
    
    def __init__(self, t = None, bw = None, window_type = 'flat', metric = 'euclidean'):
        
        self.t = t
        self.bw = bw
        self.window_type = window_type
        self.metric = metric
            
    def train(self, data):
        
        if self.t is None:
            self.t = estimate_bandwidth(data)
        if self.bw is None:
            self.bw = estimate_bandwidth(data)
        
        dist_mat = pairwise_distances(data, metric=self.metric)
        
        if self.window_type == 'flat':
            weight_mat = 1 * (dist_mat <= self.bw)
        else:
            weight_mat = np.exp(-dist_mat**2 / (2 * self.bw**2))
            
        P = sum(weight_mat)
        P = P[:, np.newaxis] - P
        dist_mat[dist_mat == 0] = self.t/2
        S = np.sign(P) * (1/dist_mat)
        S[dist_mat > self.t] = -1
        
        medoids = np.argmax(S, axis=0)
        
        stat_idx = []
        for i in range(len(medoids)):
            if medoids[i] == i:
                stat_idx.append(i)
        
        cluster_centers_idx_ = np.asarray(stat_idx)
        
        cluster_centers_ = data[cluster_centers_idx_]
        
        labels_ = []
        labels_val = {}
        lab = 0
        for i in cluster_centers_idx_:
            labels_val[i] = lab
            lab += 1
            
        for i in range(len(data)):
            next_med = medoids[i]
            while next_med not in cluster_centers_idx_:
                next_med = medoids[next_med]
            labels_.append(labels_val[next_med])
            
        self.cluster_centers = cluster_centers_
        self.labels = np.asarray(labels_)
        self.cluster_centers_idx = cluster_centers_idx_
        
        return self
    
    
        
        