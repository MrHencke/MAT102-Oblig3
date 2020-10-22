from __future__ import division
import pca
import scipy.io
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import pylab

data = scipy.io.loadmat("Arbeidskrav3.mat")

X1 = np.array(data['X1'])
X2 = np.array(data['X2'])

#Oppgave 1