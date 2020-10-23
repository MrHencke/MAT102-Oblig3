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

# Oppgave 1
# Preprossessering, meancenter for gjennomsnitt = 0 i hver søyle/kolonne,
# standardize for standardavvik = 1 i hver søyle/kolonne
X1 = pca.meanCenter(X1)
X1 = pca.standardize(X1)

# Navn på punkt i score plot
objNames1 = data['objNames1']
# Navn på punkt i loading plot
varNames1 = data['varNames1']


# Oppgave 2
# Standardisering i PCA er viktig fordi PCA beskriver varians i et sett.
# Når vi standardiserer dataene vi tar inn, får vi sammenlignbare resultater,
# på tvers av måleenheter og størrelsesordener

# Oppgave 3
[T, P, E] = pca.pca(X1, a=2)

plt.figure(0)
plt.title('Score plot')
plt.scatter(T[:, 0], T[:, 1])  # Datapkt for de to første prinsipalkomponentene
for label, x, y in zip(objNames1, T[:, 0], T[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, -3),
        textcoords='offset points', ha='left')

# Oppgave 4
# Forsøkene 5a og 5b er svært nære hverandre, se figur1.png

# Oppgave 5
varX1 = np.trace(np.dot(X1, X1.T))
varT = np.trace(np.dot(T, T.T))
print(varT/varX1)
# VarT/VarX1 gir 98% av variansen

# Oppgave 6
plt.figure(1)
plt.title('Loading plot')
plt.scatter(P[:, 0], P[:, 1])
for label, x, y in zip(varNames1, P[:, 0], P[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, -3),
        textcoords='offset points', ha='left')

plt.show()
