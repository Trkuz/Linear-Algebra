import numpy as np
import matplotlib.pyplot as plt
import copy
import math

#1 d
def compare():
    K = np.random.randn(4,4)
    Q, _ = np.linalg.qr(K)
    Qinv = np.linalg.inv(Q)
    print(np.round(np.abs(Q.T@Q)),'\n','\n', np.round(np.abs(Q@Q.T)),'\n','\n', np.round(np.abs(Qinv@Q)),'\n','\n', np.round(np.abs(Q@Qinv)),'\n','\n', np.eye(4))

#2
def GS():
    K = np.random.randn(4,4)
    Q = np.zeros(K.shape)
    for i in range(0, K.shape[1]):
        v = K[:,i]

        for j in range(0,i):
            proj = (np.dot(v, Q[:, j]) / np.dot(Q[:, j], Q[:, j])) * Q[:, j] #znajdujemy rzut wetora v(tego którego ortogonalizujemy na wetkor Q)
            v -= proj  #odejmujemy rzut od wektora v do daje nam wektor prostopadły do Q

        v = v/np.linalg.norm(v)
        Q[:,i] = v

        print(Q)
        print(np.linalg.qr(K))

#3
def idk():
    K = np.random.randn(6,6)
    U, _ = np.linalg.qr(K)


    for i in range(U.shape[0]):
        U[:, i] = U[:, i] * (i+10)

    Q, R = np.linalg.qr(U)
    print(np.round(R))

def err(A):
    def OldSchoolInv(A):
        if A.shape[0] != A.shape[1]:
            raise ("Matrix must be square!")
        if np.linalg.matrix_rank(A) != min(A.shape[0], A.shape[1]):
            raise ("Matrix must be of full rank!")

        minor = np.zeros(A.shape[0] * A.shape[1]).reshape(A.shape)
        grid = np.zeros(A.shape[0] * A.shape[1]).reshape(A.shape)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                D = np.delete(copy.deepcopy(A), i, 0)
                D =np.delete(copy.deepcopy(D), j, 1)

                det = np.linalg.det(D)
                minor[i, j] = det
                grid[i, j] = pow(-1, (i + j + 2))

        A = A - 0.01
        alg = grid * minor

        final = alg.T * (1 / np.linalg.det(A))
        return final

    old = OldSchoolInv(A)
    Q,R = np.linalg.qr(A)
    Rinv = OldSchoolInv(R)
    new = Rinv@Q.T

    old_eye = A@old
    new_eye = A@new

    old_err = np.sqrt(np.sum(np.square(old_eye - np.eye(A.shape[0]))))
    new_err = np.sqrt(np.sum(np.square(new_eye - np.eye(A.shape[0]))))

    return old_err, new_err

    # plt.figure(figsize=(6,6))
    # plt.bar(5, old_err, width= 4)
    # plt.bar(10, new_err, width= 4)
    # plt.title(f"Błąd w obliczneniach odwrotności(macierz o wymiarach: {A.shape[0]}x{A.shape[0]}")
    # plt.ylabel("Odl. euklidesowa do macierzy jednostkowej")
    # plt.xticks([5,10], ['Staromodna metoda','Rozkład QR'])
    #
    # plt.show()

err(A = np.random.randn(5,5))

def method_comparison():

    Results30 = np.zeros((100,2))
    Results5 = np.zeros((100,2))


    for i in range(200):
        if i <= 99:
            A = np.random.rand(5,5)
            Results5[i,0], Results5[i,1] = err(A)
        else:
            A = np.random.rand(30,30)
            Results30[i-100, 0], Results30[i-100, 1] = err(A)

    print(Results30)
    ones = np.ones(Results5[:, 0 ].shape)
    # print(ones)
    # print(Results5[:,0].shape)
    fig, ax = plt.subplots(1,2, figsize = (6,8))

    ax[0].bar(10, np.mean(Results5[:,0]), width = 5)
    ax[0].bar(20, np.mean(Results5[:, 1]), width=5)
    ax[0].plot(ones*10,Results5[:,0], 'o', color = 'black')
    ax[0].plot(ones*20,Results5[:, 1], 'o', color = 'black')
    ax[0].set_xticks([10,20], ["Staromodna metoda", "Rozkład QR"])
    ax[0].set_title("Błąd w obliczeniach w odwrtoności (Macierz o wymiarach 5x5)")
    ax[0].set_ylabel("Odl. euklidesowa od maciery jednostkowej")

    ax[1].bar(10, np.mean(Results30[:, 0]), width=5)
    ax[1].bar(20, np.mean(Results30[:, 1]), width=5)
    ax[1].plot(ones * 10, Results30[:, 0], 'o', color='black')
    ax[1].plot(ones * 20, Results30[:, 1], 'o', color='black')
    ax[1].set_xticks([10, 20], ["Staromodna metoda", "Rozkład QR"])
    ax[1].set_title("Błąd w obliczeniach w odwrtoności (Macierz o wymiarach 30x30)")
    ax[1].set_ylabel("Odl. euklidesowa od maciery jednostkowej")
    plt.show()

#6
def properties(i):
    K = np.random.randn(i,i)
    Q,R = np.linalg.qr(K)
    indcued = max([np.round(np.linalg.norm(Q[:,i])) for i in range(K.shape[0])])
    print(indcued)
    Qnorm = np.sqrt(np.sum(np.square(Q)))
    print(round(Qnorm/math.sqrt(K.shape[0])))

    V = np.random.randn(i,1)
    print(np.linalg.norm(V))
    print(np.linalg.norm(Q@V))


def properties2():
    N = np.random.randn(10,4)

    Q,R = np.linalg.qr(N, 'complete')
    print(R)
    #print(R[0:4,0:4])
    inv1 = np.linalg.inv(R[0:4,0:4])
    inv2 = np.linalg.pinv(R)
    #print(inv1)
    #print(inv2)

