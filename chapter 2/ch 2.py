import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_norm(x):

    norm = 0
    r = x.shape[0]
    c = x.shape[1]

    for i in range(r):
        for j in range(c):
            norm += pow(x[i][j],2)

    return math.sqrt(norm)


def versor(x):
    r = x.shape[0]
    c = x.shape[1]
    v = np.zeros((r,c))
    norm = calculate_norm(x)

    for i in range(r):
        for j in range(c):
            v[i][j] = x[i][j]/norm

    return v

vec = np.array([[1,2,3],
                [3,4,5]])


def cos_tam(x, l):
    v = np.zeros((x.shape[0], x.shape[1]))
    v_0 = versor(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v[i][j] = v_0[i][j] * l

    return v



def transpose(x):
    r = np.shape(x)[0]
    c = np.shape(x)[1]
    t = np.zeros((c,r))

    for i in range(c):
        for j in range(r):
            t[i][j] = x[j][i]

    return t

vec1 = np.array([[1,2,3]])
vec2 = np.array([[4,5,6]])


def ortogonal(r,t):
    t_1 = r * (np.dot(r, t.T)/np.dot(r, r.T)) #dobrze
    print(t_1)
    t_2 = t - t_1
    print(t[0][1])
    color = ['r', 'g', 'b', 'y']
    plt.quiver([0,0,0,0], [0,0,0,0], [t[0][0], t_1[0][0],t_2[0][0], r[0][0]], [t[0][1], t_1[0][1],t_2[0][1],r[0][1]],
               scale = 1, angles = 'xy', scale_units = 'xy', linestyle = ['-','--','--','-'],linewidth=1, fc = 'none', ec = color)
    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    return t_1, t_2, np.dot(t_1, t_2.T)

def draw(r,t):
    t_1 = r * (np.dot(r, t.T) / np.dot(r, r.T))  # dobrze
    print(t_1)
    t_2 = t - t_1
    color = ['r', 'g', 'b']
    plt.quiver([0,t_1[0][0],0], [0,t_1[0][1],0], [t[0][0],t_2[0][0], r[0][0]], [t[0][1],t_2[0][1],r[0][1]],
               scale = 1, angles = 'xy', scale_units = 'xy', linestyle = ['-','--','-'],linewidth=1, fc = 'none', ec = color)

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def sanity_test(r,t):
    t_1 = r * (np.dot(r, t.T)/np.dot(r, r.T))
    print(t_1)
    t_2 = t - t_1
    print(np.dot(t_1,t_2.T))

    t_1a = r * (np.dot(r, t.T)/np.dot(t, t.T))
    t_2a = t - t_1a
    print(np.dot(t_1a, t_2a.T))







