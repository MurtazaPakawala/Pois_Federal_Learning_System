#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:37:00 2021

@author: murtaz
"""

import matplotlib.pyplot as plt
import numpy as np

data_np=np.load('tp_100_benign_acc.npy')
data_p=np.load('tp_100_mal_acc.npy')
training=np.arange(24)
print(training)

plt.plot(training,data_np)
plt.plot(training,data_p)
plt.show()
