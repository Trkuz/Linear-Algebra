import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def z1():
    A = np.random.randn(6,6)
    B = np.random.randn(1,6)
    #print(A)

    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    U1,S1,Vt1 = np.linalg.svd(A)

    print(np.linalg.norm(B))
    print(U1)
    print(np.linalg.norm(U1@B.T))


def z4():

    U,_ = np.linalg.qr(np.random.randn(10,10))
    V,_ = np.linalg.qr(np.random.randn(6,6))

    K = np.linspace(42,1,6)
    S = np.zeros((10,6))
    np.fill_diagonal(S, K )

    A = U@S@V

    _, axs = plt.subplots(1, 4, figsize=(12, 6))

    axs[0].imshow(A, aspect='equal', cmap='gray')
    axs[0].set_title(f'A (cond={np.linalg.cond(A):.3f})')

    axs[1].imshow(U, aspect='equal', cmap='gray')
    axs[1].set_title(f'U (cond={np.linalg.cond(U):.3f})')

    axs[2].imshow(S, aspect='equal', cmap='gray')
    axs[2].set_title(f'$\Sigma$ (cond={np.linalg.cond(S):.3f})')

    axs[3].imshow(V, aspect='equal', cmap='gray')
    axs[3].set_title(f'V$^T$ (cond={np.linalg.cond(V):.3f})')

    plt.tight_layout()
    plt.savefig('rys14.4.png', dpi=300)
    plt.show()

def z5():
    m = 40
    n = 30

    k = int((m + n) / 4)
    X, Y = np.meshgrid(np.linspace(-3, 3, k), np.linspace(-3, 3, k))
    g2d = np.exp(-(X ** 2 + Y ** 2) / (k / 8))

    A = convolve2d(np.random.randn(m, n), g2d, mode='same')

    U, s, Vt = np.linalg.svd(A)
    S = np.zeros(np.shape(A)) #macierz wartośći osobliwych
    np.fill_diagonal(S, s)

    # _, ax = plt.subplots(1,4, figsize = (6,8))
    # ax[0].imshow(A, cmap= 'gray')
    # ax[0].set_title("A")
    # ax[1].imshow(U, cmap= 'gray')
    # ax[0].set_title("U")
    # ax[2].imshow(S, cmap= 'gray')
    # ax[0].set_title("sigma")
    # ax[3].imshow(Vt, cmap= 'gray')
    # ax[0].set_title("Vt")

    #plt.show()

    total = 0
    res = np.zeros((np.shape(s)[0],2))
    for x in s:
        total = total + x*x

    # plt.figure(figsize = (6,10))
    for i, x in enumerate(s):

        res[i,0] = i
        res[i,1] = ((x*x)/total) *100

    #
    # plt.scatter(res[:,0], res[:,1], marker = 's', s = 20, color = 'red')
    # plt.plot(res[:,0], res[:,1], lw = 1, color = 'red')
    # plt.xlabel("Procent wyjaśnaniej wariancji")
    # plt.ylabel("Numer składowej( wartości osobliwej)")
    # plt.title("Wykres osypiska")
    # plt.show()

    _, ax = plt.subplots(2, 5, figsize=(6, 8))
    sum = np.zeros_like(A)
    for i in range(4):
        L = np.expand_dims(U[:,i],1)@np.expand_dims(s[i], [0,1])@np.expand_dims(Vt[i,:],0)
        ax[0,i].imshow(L, cmap = 'gray')
        ax[0, i].set_title(f"L: {i}")
        sum +=L
        ax[1,i].imshow(sum, cmap = 'gray')
        ax[1, i].set_title(f"L: 0 : {i}")

    ax[0,4].imshow(A, cmap='gray')
    ax[0, 4].set_title("A")

    for i in ax.flatten(): i.set_axis_off()
    plt.tight_layout()
    plt.show()


def MP():
    A = np.random.randn(5, 3) @ np.random.randn(3, 5)

    U, s, Vt = np.linalg.svd(A)

    tol = np.finfo(float).eps * np.max(A.shape) #najwmiejsza możliwa liczba która po dodaniu do 1 nie da 1 * wieszy wimatr macierzy

    # odwracam sigmy powyżej progu
    sInv = np.zeros_like(s)
    sInv[s > tol] = 1 / s[s > tol] #odwracamy tylko liczby powyżej progu tolerancji

    S = np.zeros_like(A)

    np.fill_diagonal(S, sInv)
    Apinv = Vt.T @ S @ U.T

    return Apinv

def z7():
    A = np.random.randn(5,3)
    print(np.linalg.inv(A.T@A)@A.T)
    print(np.linalg.pinv(A))

def z8():
    M = np.array([
        [-1,1],
        [-1,2]
    ])
    evals, evecs = np.linalg.eig(M)
    l = evals[1]  # dla wygody zapisuję lambda1 w osobnej zmiennej
    v = evecs[:, [1]]  # dla wygody zapisuję wektor własny związany z lambda1 w osobnej zmiennej


    LHS = M @ v
    RHS = l * v

    # wyświetlam obie strony (dla wygodny w postaci wektorów wierszowych)
    print(LHS.T)
    print(RHS.T)

    # pinv(v)
    vPinv = np.linalg.pinv(v)

    # sprawdzam
    vPinv @ v

    # 1
    LHS = vPinv @ M @ v
    RHS = l * vPinv @ v

    print(LHS)
    print(RHS)

    #2
    LHS = M @ v @ vPinv
    RHS = l * v @ vPinv

    print(LHS), print(' ')
    print(RHS)