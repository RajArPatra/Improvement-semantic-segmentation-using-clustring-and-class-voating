# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:11:23 2020

@author: kb
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from clustering import QuickShift

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
X, y = noisy_circles
X = StandardScaler().fit_transform(X)

model = QuickShift(window_type="flat")

t0 = time.time()
model.train(X)
t1 = time.time()
y_pred = model.labels.astype(np.int)