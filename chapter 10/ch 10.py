import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

#1
def LU():
    AR = np.array([np.random.randn(100,100) ]*1000)
    t1 = time.time()
    i=0
    for element in AR:
        _,L,U = sc.linalg.lu(element)
    t2 = time.time()
    print(t2-t1)

#2
def visualize():
    A = np.random.randn(6,3)
    B = np.random.randn(3,8)

    K = A@B

    _,L,U = sc.linalg.lu(K)

    fig, ax = plt.subplots(1,3,figsize = (6,8))

    ax[0].imshow(K, cmap='gray')
    ax[0].set_title("A, rząd = 3")

    ax[1].imshow(L, cmap='gray')
    ax[1].set_title("L, rząd = 6")

    ax[2].imshow(U, cmap='gray')
    ax[2].set_title("U, rząd = 3")


    plt.show()

#3
def determinant():
    A = np.random.randn(5,5)
    det = np.linalg.det(A)
    P,L,U = sc.linalg.lu(A)

    k = 1
    for i in range(np.shape(U)[0]):
        for j in range(np.shape(U)[1]):
            if i == j:
                k *= U[i,j]

    k *=np.linalg.det(P)
    print(det)
    print(k)

def equation():
    A = np.random.randn(4,4)

    P,L,U = sc.linalg.lu(A)
    Ainv = np.linalg.inv(U)@np.linalg.inv(L)@P.T


    print(np.round(A@Ainv))

def idk():
    A = np.random.randn(4,4)
    _,L,U = sc.linalg.lu(A)

    print(A.T@A)
    print(U.T@L.T@L@U)
