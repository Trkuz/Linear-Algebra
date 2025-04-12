import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sympy as sym



numcourses = np.array([[13, 4, 12, 3, 14, 13, 12, 9, 11, 7, 13, 11, 9, 2, 5, 7, 10, 0, 9, 7]]).T  # dane niezależne (X)
happiness = np.array([[70, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 70]]).T  # dane zależne (y)

#1
def illustrate(numcourses,happiness):

    X = np.hstack((np.ones(numcourses.shape),numcourses))#
    weights = np.linalg.inv(X.T@X)@X.T@happiness
    predict = X@weights
    residues = predict - happiness

    plt.plot(residues, predict, 'o', markersize = 8, color = 'black' )
    plt.xlabel('Reszty')
    plt.ylabel('Wartości przewidywane przez model')
    plt.title('r = 0.00000000000000092748')
    plt.show()

#2
def ortogonality(numcourses,happiness):
    #czyli skoro wektor reszt jest ortogonalny do przestrzeni kolumnowej macierzy, tak samo jak jądro
    #lewostronne macierzy, to znaczy że wektor reszt należy do przestrzeni wierszowej macierzy
    #czyli tym samymy należy do jądra lewostronnego macierzy

    X = np.hstack((np.ones(numcourses.shape), numcourses))
    weights = np.linalg.inv(X.T @ X) @ X.T @ happiness
    predict = X @ weights
    residues = predict - happiness

    J = sc.linalg.null_space(numcourses.T)
    nullspaceAugment = np.hstack((J, residues.reshape(-1, 1)))

    print(f'dim(  N(X)    ) = {np.linalg.matrix_rank(J)}')
    print(f'dim( [N(X)|r] ) = {np.linalg.matrix_rank(nullspaceAugment)}')

#3

def VSLU(numcourses,happiness):

    X = np.hstack((np.ones(numcourses.shape), numcourses))  #
    weights1 = np.linalg.inv(X.T @ X) @ X.T @ happiness
    predict1 = X @ weights1
    residues1 = predict1 - happiness

    Q,R = np.linalg.qr(X)

    weights2 = np.linalg.inv(R)@ (Q.T@happiness)
    predict2 = X @ weights2
    residues2 = predict2 - happiness

    tmp = (Q.T @ happiness).reshape(-1, 1)
    Raug = np.hstack( (R,tmp) )# dopisyujemy R z lewej strony
    Raug_r = sym.Matrix(Raug).rref()[0] #0 oznacza że uzyskujemy już postać zredukowną, więc ostatnia kolumna to macierz odwrotna
    weights3 = np.array(Raug_r[:, -1])
    predict3 = X @ weights3
    residues3 = predict3 - happiness


#4
def outliers(numcourses,happiness):
    k = np.squeeze(numcourses)
    fig, ax = plt.subplots(1, 3, figsize=(18, 18))

    X = np.hstack((np.ones(numcourses.shape), numcourses))  #
    weights1 = np.linalg.inv(X.T @ X) @ X.T @ happiness
    predict1 = X @ weights1
    residues1 = predict1 - happiness


    ax[0].scatter(k, happiness,marker = 's', s=8, color='black', label = 'Rzeczywiste Dane')
    ax[0].scatter(k, predict1,marker = 'o', s=8, color = 'gray', label='Przewidywania')
    ax[0].plot(numcourses, predict1, color='gray', lw=1)
    for x, y_true, y_pred in zip(numcourses, happiness, predict1):
        ax[0].plot([x, x], [y_true, y_pred], '--', color='gray', zorder=10)
    #ax[0].plot(k, list(pairs1),'--', color = 'gray', zorder = 10)
    ax[0].set_title("SSE = 8958.78")
    ax[0].set_xlabel("Liczba kursów w których brała udział dana osoba")
    ax[0].set_xlabel("Ogólne zadowlenie z życia")
    ax[0].legend()

    happiness = np.array([[170, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 70]]).T  # dane zależne (y)

    weights2 = np.linalg.inv(X.T @ X) @ X.T @ happiness
    predict2 = X @ weights2
    residues2 = predict2 - happiness



    ax[1].scatter(numcourses, happiness,marker = 's', s=8,  color='black', label='Rzeczywiste Dane')
    ax[1].scatter(numcourses, predict2,marker = 'o', s=8, color = 'gray', label='Przewidywania')
    ax[1].plot(numcourses, predict2, color='gray', lw=1)
    for x, y_true, y_pred in zip(numcourses, happiness, predict2):
        ax[1].plot([x, x], [y_true, y_pred], '--', color='gray', zorder=10)
    ax[1].set_title("SSE = 10328.89")
    ax[1].set_xlabel("Liczba kursów w których brała udział dana osoba")
    ax[1].set_xlabel("Ogólne zadowlenie z życia")
    ax[1].legend()

    happiness = np.array([[70, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 170]]).T  # dane zależne (y)

    weights3 = np.linalg.inv(X.T @ X) @ X.T @ happiness
    predict3 = X @ weights3
    residues3 = predict3 - happiness


    ax[2].scatter(numcourses, happiness,marker = 's', s=8, color='black', label='Rzeczywiste Dane')
    ax[2].scatter(numcourses, predict3, marker = 'o',s=8, color = 'gray', label='Przewidywania')
    ax[2].plot(numcourses, predict3, color='gray', lw=1)
    for x, y_true, y_pred in zip(numcourses, happiness, predict2):
        ax[2].plot([x, x], [y_true, y_pred], '--', color='gray', zorder=10)
    ax[2].set_title("SSE = 246338.57")
    ax[2].set_xlabel("Liczba kursów w których brała udział dana osoba")
    ax[2].set_xlabel("Ogólne zadowlenie z życia")
    ax[2].legend()

    plt.show()

#5

def costam():
    n = 6
    X = np.random.randn(n, n)
    Y = np.eye(n)

    Xinv1 = np.zeros_like(X)


    for coli in range(n):
        Xinv1[:, coli] = np.linalg.inv(X.T @ X) @ X.T @ Y[:, coli]

    # to samo, ale bez użycia pętli
    Xinv2 = np.linalg.inv(X.T @ X) @ X.T @ Y

    # obliczanie odwrotności za pomocą inv()
    Xinv3 = np.linalg.inv(X)

    # wizualizacja
    _, axs = plt.subplots(1, 3, figsize=(10, 6))

    axs[0].imshow(Xinv1 @ X, cmap='gray')
    axs[0].set_title('Metoda najmniejszych\nkwadratów:\nkolumna po kolumnie')

    axs[1].imshow(Xinv2 @ X, cmap='gray')
    axs[1].set_title('Metoda najmniejszych\nkwadratów:\ncała macierz naraz')

    axs[2].imshow(Xinv3 @ X, cmap='gray')
    axs[2].set_title('Funkcja inv()\n')

    for a in axs: a.set(xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig('rys11.8.png', dpi=300)
    plt.show()


costam()





