import matplotlib.animation
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert


#1
def inve():
    A = np.random.randn(16).reshape((4,4))
    Ainv = np.linalg.inv(A)
    print(A == np.linalg.inv(Ainv))

#2
def manual_inversion(A):
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
                D = np.delete(copy.deepcopy(D), j, 1)

                det = np.linalg.det(D)
                minor[i, j] = det
                grid[i, j] = pow(-1, (i + j + 2))

        A = A - 0.01
        alg = grid * minor

        final = alg.T * (1 / np.linalg.det(A))
        return final

    fig, ax = plt.subplots(2,3,figsize = (6,8))
    ax[0,0].imshow(minor, cmap = 'gray')
    ax[0,0].set_title("Macierz minorów")

    ax[0,1].imshow(grid, cmap = 'gray')
    ax[0,1].set_title("Siatka")

    ax[0,2].imshow(alg, cmap = 'gray')
    ax[0,2].set_title("Macierz dopełnień algebraicznych")

    ax[1, 0].imshow(final, cmap = 'gray')
    ax[1, 0].set_title("Macierz dołączona")

    ax[1, 1].imshow(np.linalg.inv(A), cmap = 'gray')
    ax[1, 1].set_title("np.linalg.inv")

    ax[1, 2].imshow(A@np.linalg.inv(A), cmap = 'gray')
    ax[1, 2].set_title("A@Ainv")
    plt.show()

A = np.random.randn(16).reshape((4,4))

#4
def right_inverse(H):

    inverse = H.T@(np.linalg.inv(H@H.T))

    fig, ax = plt.subplots(2,2,figsize = (8,8))

    ax[0,0].imshow(H)
    ax[0,0].set_title('Macier szeroka')

    ax[0,1].imshow(inverse)
    ax[0,1].set_title('Odwrotność prawostronna')

    ax[1,0].imshow(H@inverse)
    ax[1,0].set_title('H@inverse')

    ax[1,1].imshow(inverse@H)
    ax[1,1].set_title('inverse@H')

    for i in ax.flatten(): i.axis('off')

    plt.show()

#5
def pseudo(H):

    if H.shape[0] < H.shape[1]:
        inv = H.T@(np.linalg.inv(H@H.T))
        return np.linalg.pinv(H) == inv

    elif H.shape[0] > H.shape[1]:
        inv = (np.linalg.inv(H.T@H))@H.T
        return np.linalg.pinv(H) == inv

    return np.linalg.pinv(H) == np.linalg.inv(H)


#6
def operations(A,B):
    C = np.random.randn(16).reshape((4, 4))
    D = np.random.randn(16).reshape((4, 4))


    AA = np.linalg.inv(A.T@A)
    AB = np.linalg.inv(A@B)
    ABCD = np.linalg.inv(A@B@C@D)
    A1B1 = np.linalg.inv(A)@np.linalg.inv(B)
    B1A1 = np.linalg.inv(B)@np.linalg.inv(A)
    AA1 = np.linalg.inv(A)@np.linalg.inv(A.T)


    D1C1B1A1 = np.linalg.inv(D)@np.linalg.inv(C)@np.linalg.inv(B)@np.linalg.inv(A)

    dist1 = AB - A1B1
    dist1 = np.sqrt(np.sum(np.square(dist1)))

    dist2 = AB - B1A1
    dist2 = np.sqrt(np.sum(np.square(dist2)))

    dist3 = ABCD - D1C1B1A1
    dist3 = np.sqrt(np.sum(np.square(dist3)))

    dist4 = AA - AA1
    dist4 = np.sqrt(np.sum(np.square(dist4)))

    print(f"Odległość pomiędzy macierzami (AB)^-1 i (A^-1)(B^-1) to {dist1}")
    print(f"Odległość pomiędzy macierzami (AB)^-1 i (B^-1)(A^-1) to {dist2}")
    print(f"Odległość pomiędzy macierzami (ABCD)^-1 i (D^-1)(C^-1)(B^-1)(A^-1) to {dist3}")
    print(dist4)
#8
def reverse_animation():
    T = np.array([
        [1,0.5],
        [0,0.5]
    ])

    angle = np.linspace(0, 2*np.pi, 20)
    points = np.vstack((np.sin(angle), np.cos(angle)))

    plt.figure(figsize=(6,6))

    plt.plot(points[0,:], points[1,:], 'ko', color = 'red', label = 'Punkty')


    points1 = T@points
    plt.plot(points1[0,:], points1[1,:], 's', color = 'blue', label = 'po przekształceniu')


    points2 = np.linalg.pinv(T) @ points1

    plt.plot(points2[0, :], points2[1, :], 'x', color='blue', label='po zastosowaniu macierzy odwrotnej')
    plt.legend()
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()

#9
def Hilbert(size):
    H = np.zeros(size*size).reshape((size,size))
    for i in range(size):
        for j in range(size):
            print(i+j+1)
            H[i,j] = 1/(i+j+1)


    k = np.arange(1, size + 1).reshape(1, -1)
    H2 = 1/(k.T + k -1)

    fig, ax = plt.subplots(1,3,figsize = (6,12))

    ax[0].imshow(H, cmap = 'gray')
    ax[0].set_title("Macierz Hilberta")

    ax[1].imshow(np.linalg.inv(H), cmap = 'gray')
    ax[1].set_title("Odwrotność macierzy Hilberta")

    ax[2].imshow(np.linalg.inv(H)@H, cmap = 'gray')
    ax[2].set_title("Iloczyn")

    plt.show()

    return H

#Hilbert(5)

def compare():
    Sizes = np.arange(3, 13)
    Distances = np.zeros((10,2))
    Factor = np.zeros((10,2))


    for i in range(10):
        H = hilbert(i+3)
        Hinv = np.linalg.inv(H)
        HHi = H@ Hinv
        err = HHi - np.eye(i+3)
        Distances[i,0] = np.sqrt(np.sum(err**2))
        Factor[i,0] = np.linalg.cond(H)

        R = np.random.rand((i+3)*(i+3)).reshape((i+3,i+3))
        Rinv = np.linalg.inv(R)
        IR = R@Rinv
        err = IR - np.eye(i + 3)
        Distances[i,1] = np.sqrt(np.sum(err**2))
        Factor[i,1] = np.linalg.cond(R)

    Distances = np.log(Distances)
    Factor = np.log(Factor)

    print(Sizes)
    print(Distances)

    fig, ax = plt.subplots(1,2,figsize = (14,6))

    ax[0].plot(Sizes, Distances[:,0], 'ko', color = 'red', label = 'Macierz hilberta')
    ax[0].plot(Sizes, Distances[:,1], 's', color = 'blue', label = 'Macierze wartości losowych')
    ax[0].set_title('Odległość od macierzy jednostkowej')
    ax[0].set_xlabel("Rozmar macierzy")
    ax[0].set_ylabel("Logarytm z odległości eklidesowej")
    ax[0].legend()


    ax[1].plot(Sizes, Factor[:,0], 'ko', color = 'red', label = 'Macierz hilberta')
    ax[1].plot(Sizes, Factor[:,1], 's', color = 'blue', label = 'Macierze wartości losowych')
    ax[1].set_title('Współczynnik uwarunkowania macierzy')
    ax[1].set_xlabel("Rozmar macierzy")
    ax[1].set_ylabel("Logarytm z współczynnika uwarunkowania")
    ax[1].legend()


    for x in ax.flatten(): x.set_aspect('equal', adjustable= 'box')

    plt.axis('scaled')
    plt.tight_layout()
    plt.show()

compare()