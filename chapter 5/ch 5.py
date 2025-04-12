import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import plotly.graph_objects as go


def element(x,y):
    A = np.arange(12).reshape(3,4)
    print(f" Element na poyzycji: ({x,y}),to: {A[x-1,y-1]}")
def draw_matrix(C,reverse = False):

    if reverse:
        slices = [C[i:i + 5, j: j + 5] for j in [0, 5] for i in [0, 5]]


    C1 = C[0:5:1,0:5:1]
    norm = matplotlib.colors.Normalize(C.min(), C.max()) #create colormap for the matrix


    plt.figure(1)
    plt.imshow(C, cmap = 'gray', norm = norm)
    plt.title("Oryginalna macierz")
    plt.vlines(x = 4.5, ymin = 9.5, ymax= -0.5, colors = 'white', linestyles= 'dashed')
    plt.hlines(y = 4.5, xmin = -0.5, xmax= 9.5, colors = 'white', linestyles= 'dashed')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            text = plt.text(j, i, C[i, j], ha='center', va='center', color = 'gray')

    plt.figure(2)
    plt.imshow(C1, cmap = 'gray', norm = norm)
    plt.title("Podmacierz")
    for i in range(C1.shape[0]):
        for j in range(C1.shape[1]):
            text = plt.text(j, i, C1[i, j], ha='center', va='center', color= 'gray')


    if reverse:
        plt.figure(3)
        m_1 = np.concatenate([slices[3], slices[2]], axis = 0)
        m_2 = np.concatenate([slices[1], slices[0]], axis = 0)
        m = np.concatenate([m_1,m_2], axis = 1)
        plt.imshow(m, cmap = 'gray', norm = norm)
        plt.title("Macierz po zmianie kolejności bloków")
        plt.vlines(x=4.5, ymin=9.5, ymax=-0.5, colors='white', linestyles='dashed')
        plt.hlines(y=4.5, xmin=-0.5, xmax=9.5, colors='white', linestyles='dashed')
        ixc = list(itertools.chain.from_iterable(m.tolist()))

        k = 0
        for i in range(0,10):
            for j in range(0,10):
                plt.text(j,i, ixc[k], ha = 'center', va = 'center', color = 'lightgray')
                k += 1


    plt.show()
C = np.arange(100).reshape(10,10)

def check_equal():
    A = np.random.normal(0.0,1.0,(3,4))
    B = np.random.normal(0.0,1.0,(3,4))
    scalar = random.uniform(-999,999)
    first = (A + B) * scalar
    second = (A * scalar) + (B * scalar)
    third = (scalar * A) + (scalar * B)
    a = first == second
    b = first == third

    return a == b

def matrix_multiply(A,B):
    if np.shape(A)[1] != np.shape(B)[0]:
        raise ValueError("Matrices of that shaped can't be multiplied! The shape must be: MxN @ NxK")
        exit(0)

    M = np.zeros(np.shape(A)[0]*np.shape(B)[1]).reshape(np.shape(A)[0], np.shape(B)[1])
    for i in range(np.shape(A)[1]-1):
        for j in range(np.shape(B)[0]-1):

            M[i,j] = np.dot(A[i,:], B[:,j])

    return M

A = np.arange(12).reshape(3,4)
B = np.arange(12).reshape(4,3)

L = np.random.normal(0.0,1.0,(2,2))
I = np.random.normal(0.0,1.0,(2,2))
V = np.random.normal(0.0,1.0,(2,2))
E = np.random.normal(0.0,1.0,(2,2))
# print((L@I@V@E).T)
#
# print((E.T)@(V.T)@(I.T)@(L.T))
#
# print((L.T)@(I.T)@(V.T)@(E.T))
#

def check_symmetrical(A):
    if np.shape(A)[0] != np.shape(A)[1]:
        return False

    if np.array_equal(A.T,A):
        return True
    return False

A = np.arange(9).reshape(3,3)
B = np.eye(5)

print(check_symmetrical(A))


def make_symmetrical(A):
    B = A.T


    return (A+B)

print(make_symmetrical(A))

print(check_symmetrical(make_symmetrical(A)))

def visualize_norm(A,B):
    M = np.concatenate((A,B), axis = 1)
    p = []
    for i in range(100):
        S = [random.uniform(-999,999),random.uniform(-999,999)]
        Sa = np.array(S).T
        p.append(M@Sa)

    fig = go.Figure(data = [go.Scatter3d(x = [x[0] for x in p],
                                         y = [x[1] for x in p],
                                         z = [x[2] for x in p],
                                         mode = 'markers')])


    fig.show()

X = np.random.normal(0.0,1.0, 4).reshape(1,4).T
Y = np.arange(4).reshape(1,4).T
#visualize_norm(X,Y)

A = np.ones(16).reshape(4,4)
D = np.diag([1,4,9,16])
F = np.diag([1,2,3,4])

# print(A@D)
# print(D@A)
#
# print(F@A@F)

print(D@F)
print(D*F)


