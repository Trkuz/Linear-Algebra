import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import scipy as sp

from main11 import create_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
df = pd.read_csv(url,sep =',', encoding='unicode_escape')

#1
def compare_eig():
    A = np.random.randn(5,5)
    Ainv = np.linalg.inv(A)
    print(np.linalg.eig(A)[1], "\n","\n",np.linalg.eig(Ainv)[1])
    print(np.linalg.eig(A)[0], "\n", "\n",print(1/np.linalg.eig(A)[0]))

#2
def p1():
    A = np.random.randn(2,2)
    A = A.T@A

    _, eigvec = np.linalg.eig(A)
    v1 = np.array(eigvec[:,0], ndmin = 2).T
    v2 = np.array(eigvec[:,1], ndmin = 2).T
    A1 = np.dot(A,v1)
    A2 = np.dot(A,v2)

    plt.figure(figsize=(6,8))
    plt.plot([0, v1[0,0]],[0, v1[1,0]], color = 'green')
    plt.plot([0, v2[0,0]],[0, v2[1,0]], color = 'black')
    plt.plot([0, A1[0,0]],[0, A1[1,0]], ls='--', color = 'violet', zorder = 10)
    plt.plot([0, A2[0,0]],[0, A2[1,0]], ls='--', color = 'red', zorder = 10)
    plt.xlim(None, None)
    plt.ylim(None, None)
    plt.grid()
    plt.show()

#3
def z3():
    A = np.random.randn(10,10)

    Asym = A.T@A
    _,eigvec = np.linalg.eigh(Asym)
    Adiagshuffle = eigvec.T@Asym@eigvec
    Adiagorgin = copy.deepcopy(Adiagshuffle)
    AdiagUp = copy.deepcopy(Adiagshuffle)
    AdiagLow = copy.deepcopy(Adiagshuffle)

    vals_l = copy.deepcopy(np.diag(Adiagshuffle))
    vals_u = copy.deepcopy(np.diag(Adiagshuffle))

    np.sort(vals_l)

    arr_l = vals_l[0:2]
    arr_u = vals_l[-2:]

    vals_l[0:2] = arr_l[::-1]
    vals_u[-2:] = arr_u[::-1]

    vals = copy.deepcopy(np.diag(Adiagshuffle))
    np.random.shuffle(vals)
    np.fill_diagonal(Adiagshuffle, vals)
    np.fill_diagonal(AdiagUp, vals_u)
    np.fill_diagonal(AdiagLow, vals_l)

    Asymorig = eigvec @ Adiagorgin@np.linalg.inv(eigvec)
    Asymshuf = eigvec @ Adiagshuffle @ np.linalg.inv(eigvec)
    AsymLow = eigvec @ AdiagLow @ np.linalg.inv(eigvec)
    AsymUp = eigvec @ AdiagUp @ np.linalg.inv(eigvec)

    F1 = np.sqrt(np.sum(np.square(Asymorig-Asym)))
    F2 = np.sqrt(np.sum(np.square(Asymshuf-Asym)))
    F3 = np.sqrt(np.sum(np.square(AsymUp-Asym)))
    F4 = np.sqrt(np.sum(np.square(AsymLow-Asym)))

    plt.figure(figsize=(6,8))
    plt.bar([10,20,30,40], [F1,F2,F3,F4], width=8)
    plt.xticks([10,20,30,40],['Brak', 'Wszystkie', 'Dwie największe', 'Dwie najmniejsze'])
    plt.xlabel('Rodzaj zmiany w kolejności wartości własnych')
    plt.ylabel('Odległość Forbeniusa do oryginalnej macierzy')
    plt.title('Dokładność rekonstrukcji')
    plt.xlim(None, None)
    plt.ylim(None, None)
    plt.tight_layout()
    plt.show()

#4
def z4():
    plt.figure(figsize=(6, 6))
    V = np.zeros((123,42), dtype=complex)

    for i in range(123):
        A = np.random.randn(42,42)
        eigval, _ = np.linalg.eig(A)
        V[i,:] = eigval/math.sqrt(42)
        plt.scatter(V[i,:].real, V[i,:].imag)

    plt.xlabel('Część rzeczywista')
    plt.ylabel('Część urojona')
    plt.xlim([None,None])
    plt.ylim([None,None])
    plt.show()

#5
def z5():
    for _ in range(1000):
        A = np.random.randn(3,3)
        A = A.T@A

        egival, eigvec = np.linalg.eigh(A)
        vec1 = np.zeros_like(eigvec)

        for i,val in enumerate(egival):
            B = A - (val* np.eye(3))
            v = sp.linalg.null_space(B).T
            print(v)
            print(B, np.linalg.matrix_rank(B))
            vec1[:,i] = v

        print(vec1)
        print(eigvec)

#6
def z6():
    D = np.random.randn(4)
    Diag = np.diag(np.abs(D)) #L
    Q,_ = np.linalg.qr(np.random.randn(4,4)) #V

    A = Q@Diag@np.linalg.inv(Q)
    eigval, eigvec = np.linalg.eigh(A)
    print(np.sort(eigval))
    print(np.sort(Diag.diagonal()))
    print(eigvec)
    print(Q)


def z7(A,B):

    Anorm, _ = np.linalg.eig(A.T@A)
    Bnorm, _ = np.linalg.eig(B.T@B)

    regularizerA = np.mean(Anorm)* np.eye(4)
    regularizerB = np.mean(Bnorm)* np.eye(3)
    gamma = np.linspace(0,0.2,40)

    Results = np.zeros((40, 2))
    y = df['Rented Bike Count'].to_numpy()
    y = y.reshape((y.shape[0],1))

    for i,x in enumerate(gamma):
        B1 = np.linalg.inv(B.T@B + x*regularizerB)
        A1 = (np.linalg.inv(A.T@A + x*regularizerA))

        Beta_A = A1@A.T@y
        Beta_B = B1@B.T@y

        PredA = A@Beta_A
        PredB = B@Beta_B
        Results[i,0] = np.corrcoef(y.T,PredA.T)[0,1]**2
        Results[i,1] = np.corrcoef(y.T,PredB.T)[0,1]**2

    plt.figure(figsize = (6,8))
    plt.scatter(gamma, Results[:,0],marker = 'o', s=20,color= 'gray', label = 'Współliniowe')
    plt.plot(gamma, Results[:,0], color = 'gray', lw = 2)

    plt.scatter(gamma, Results[:, 1], marker='s', s=20, color='black', label='Oryginalne Dane')
    plt.plot(gamma, Results[:, 1], color='black', lw=2)

    plt.xlabel('Wpołczynnik regularyzacji')
    plt.ylabel('Rsquared')
    plt.legend()
    plt.show()

def z8():
    corr = np.array([
        [1,0.2,0.9],
        [0.2,1,0.3],
        [0.9,0.3,1]

    ])
    eigval, eigvec = np.linalg.eigh(corr)

    Diag = np.diag(eigval)

    Y = eigvec@np.sqrt(Diag)@np.random.randn(3,10000)
    Ydash = Y.T@eigvec@np.sqrt(Diag)

    print(np.corrcoef(Ydash))


