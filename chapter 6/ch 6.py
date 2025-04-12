import numpy as np
import matplotlib.pyplot as plt
import random
import math

def norm():
    norms = np.zeros((40,10))

    for i in range(40):
        num = random.uniform(0, 50)
        for j in range(10):

            n = np.random.normal(0.0,1.0, 100).reshape((10,10))
            n = n * num
            nnorm = np.linalg.norm(n)
            norms[i, j] = nnorm

    return norms

def forb(A,B):
    if A.shape != B.shape:
        raise Exception("The matrices mus be of the same shape!")

    return np.linalg.norm(A-B)

def find_distance():

    B = np.random.normal(0.0,1.0,49).reshape(7,7)
    s = 1
    i = 0
    while forb(A,B) > 1:
        i += 1
        s *= 0.9999999
        A = s * A
        B = s * B
        print(forb(A,B))

    return i, s, forb(A,B)

def compare(A):

    return np.linalg.norm(A) == np.sqrt(np.trace(A@A.T))

A = np.random.normal(0.0,1.0,42).reshape(6,7)

def examine():
    r = np.random.normal(0.0,1.0,100).reshape((10,10))
    n = np.linalg.norm(r)
    coef = []
    norms = []
    perc = []
    fracs = []
    for i in range(30):
        frac = (i+1)/30
        fracs.append(frac)
        r1 = r + (frac * np.identity(10) * np.linalg.norm(r))
        perc.append(((np.linalg.norm(r1) - np.linalg.norm(r))/np.linalg.norm(r)) *100)
        norms.append(np.linalg.norm(r1-r))
        coef.append(np.corrcoef(r.flatten(), r1.flatten())[1,0])

    plt.subplot(1,3,1)
    plt.scatter(fracs, perc, color = 'red')
    plt.plot(fracs, perc, linewidth=1, color = 'red')
    plt.xlim(min(fracs)  - 0.1, max(fracs) + 0.1)
    plt.ylim(min(perc) - 5, max(perc) + 5)
    plt.xlabel("Przesunięcie(jako ułamek nomry)")
    plt.ylabel("Zmiana w normie(%)")

    plt.subplot(1, 3, 2)
    plt.scatter(fracs, coef, color = 'green')
    plt.plot(fracs, coef, linewidth= 1, color = 'green')
    plt.xlim(min(fracs)  - 0.1, max(fracs) + 0.1)
    plt.ylim(min(coef) - 0.015,max(coef) + 0.015)
    plt.xlabel("Przesunięcie(jako ułamek nomry)")
    plt.ylabel("Korelacja z oryg macierzą")

    plt.subplot(1, 3, 3)
    plt.scatter(fracs, norms)
    plt.plot(fracs, norms, linewidth=1)
    plt.xlim(min(fracs)  - 0.1, max(fracs) + 0.1)
    plt.ylim(min(norms) - 1,max(norms) + 1)
    plt.xlabel("Przesunięcie(jako ułamek nomry)")
    plt.ylabel("Norma Forbeniusa")

    plt.show()

def rankk(r, rows, columns):
    A = np.random.normal(0.0,1.0, r* rows). reshape((r,rows))
    B = np.random.normal(0.0,1.0, r* columns). reshape((r,columns))

    return A@B, np.linalg.matrix_rank(A@B)


def check_rank():
    A1 = np.array([[1,1,1], [2,2,2], [3,3,3]])
    A2 = np.array([[-1,-1,-1], [-2,-2,-2], [-3,-3,-3]])

    B1 = np.array([[2,2,2], [4,4,4], [6,6,6]])
    B2 = np.array([[8,8,8], [10,10,10], [12,12,12]])

    C1 = np.array([[3,3,3], [6,6,6], [9,9,9]])
    C2 = np.array([[1.25,2.5,3.75], [1.25,2.5,3.75], [1.25,2.5,3.75]])

    return np.linalg.matrix_rank(A1 + A2 ),np.linalg.matrix_rank(B2 + B1 ),np.linalg.matrix_rank(C1 + C2)
#print(check_rank())

def rankk2(r, rows):
    A = np.random.normal(0.0,1.0, r* rows). reshape((rows,r))
    B = np.random.normal(0.0,1.0, r* rows). reshape((r,rows))



    return A@B


def vranks():
    mults = np.zeros((18,18))
    adds = np.zeros((18,18))
    for i in range(2, 16):
        A = rankk2(i, 20)
        for j in range(2,16
                       ):
            B = rankk2(j,20)
            Add = np.linalg.matrix_rank(A+B)
            Mult = np.linalg.matrix_rank(A@B)
            mults[i, j] = Mult
            adds[i, j] = Add

    print(mults)
    print(adds)
    plt.subplot(1,2,1)
    plt.imshow(mults, cmap = 'gray')
    plt.ylim(2,15)
    plt.xlim(2,15)
    plt.title("Rząd A+B")
    plt.xlabel("Rząd A")
    plt.ylabel("Rząd B")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(adds, cmap = 'gray')
    plt.ylim(2, 15)
    plt.xlim(2, 15)
    plt.title("Rząd A@B")
    plt.xlabel("Rząd A")
    plt.ylabel("Rząd B")
    plt.colorbar()

    plt.show()
#vranks()

V = np.random.normal(0,1,4).reshape((1,4))
C = np.random.normal(0,1,12).reshape((3,4))

# v2 = np.random.normal(0,1,4).reshape((1,4))
# print(C)
# print(np.linalg.matrix_rank(C))
#
# C = np.concatenate((C,V), axis = 0)
# print(C)
# print(np.linalg.matrix_rank(C))
#
# C = np.concatenate((C,v2), axis = 0)
# print(C)
# print(np.linalg.matrix_rank(C))

def show_dets():

    y = []
    x = []
    for i in range(3,31):
        for _ in range(100):
            d = 0
            K = np.random.normal(0.0, 1.0, i*i).reshape((i, i))
            K[:, -1] = K[:, 0] * 2
            d += abs(np.linalg.det(K))
        d /= 100
        x.append(i*i)
        y.append(d)

    ylog = [math.log(element)for element in y  if element >0 ]
    indexes = [index for index,element in enumerate(y) if element == 0]
    xlog = [element for element in x if x.index(element) not in indexes]

    plt.subplot(1,2,1)
    plt.scatter(xlog,ylog)
    plt.plot(xlog,ylog,linewidth = 1)
    plt.title("Wyznaczniki macierzy osobliwych")
    plt.xlabel("Licba elementów w macierzy")
    plt.ylabel("Logarytm z wyznacznika")
    plt.xlim(None, None)
    plt.ylim(None, None)

    print()

    plt.subplot(1,2,2)
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.plot(x, y, linewidth=1)
    plt.title("Wyznaczniki macierzy osobliwych")
    plt.xlabel("Licba elementów w macierzy")
    plt.ylabel("Wyznacznik")
    plt.xlim(None, None)
    plt.ylim(None, None)

    plt.show()

show_dets()