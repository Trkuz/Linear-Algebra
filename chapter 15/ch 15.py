import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import seaborn as sns
import scipy as scp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage import io, color

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx"
df = pd.read_excel(url, index_col=0, skiprows = 1)


#1
def pltt():

    header = np.array(df.columns)
    raw = df.to_numpy().T
    dates = df.axes[0].to_numpy()

    fig, ax = plt.subplots(1,2,figsize=(6,10))
    # for i in range(len(header)):
    #     plt.plot(dates, raw[i,:], label = header[i])
    # plt.legend()
    # plt.show()

    A = np.corrcoef(raw)
    B = np.cov(raw)
    pos1 = ax[0].imshow(1-A, cmap='gray')
    ax[0].set_xticks([0,1,2,3,4,5,6,7,8],header.tolist(), rotation=90)
    ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], header.tolist())
    ax[0].set_title("B) Macierz korelacji")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
                ax[0].text(i,j , np.round(A[i,j],2), ha='center', va='center', color = 'gray')

    fig.colorbar(pos1, ax=ax[0], shrink = 0.7)


    pos2 = ax[1].imshow(np.cov(raw), cmap='gray')
    ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], header.tolist(), rotation = 90)
    ax[1].set_yticks([0,1,2,3,4,5,6,7,8],header.tolist())
    ax[1].set_title("C) Macierz kowariancji")
    fig.colorbar(pos2, ax = ax[1], shrink = 0.7)


    plt.show()

def pca():
    def days_between(d1,d2):
        d1 = datetime.strptime(d1, '%Y-%m-%d')
        d2 = datetime.strptime(d2, '%Y-%m-%d')

        return abs((d2-d1).days)

    raw = df.to_numpy()
    header = np.array(df.columns)


    mean = np.mean(raw, axis =0)
    raw -= mean
    cov = np.cov(raw.T)


    U, S, Vt = np.linalg.svd(raw)

    S = np.square(S)
    s = np.argsort(S)[::-1]
    pcaval = S[s]
    pcavec = Vt.T[:,s]
    new = raw@pcavec

    total = np.sum(pcaval)
    var = pcaval/total

    fig , axs = plt.subplots(2,2,figsize = (6,30))
    ax = axs.flatten()
    ax[0].scatter([i for i in range(len(var))], var, marker = 's', s = 20, color ='blue')
    ax[0].plot([i for i in range(len(var))], var, lw =1, color = 'blue')
    ax[0].set_xlabel("Numer składowej")
    ax[0].set_ylabel("Procent wyjaśnianiej wariancji")
    ax[0].set_title('Wykres osypiska')
    dates = df.axes[0].to_numpy().astype(str)
    time = np.zeros_like(dates)

    for i,x in enumerate(dates):
        dates[i] = x[0:10]

    for i,x in enumerate(dates):
        time[i] = days_between(x, dates[0])

    ax[1].plot(time,new[:,0], label = 'Pierwsza składowa')
    ax[1].plot(time,new[:,1], label = 'Druga składowa')
    ax[1].set_xticks([0, 100, 200, 300, 400, 500])
    ax[1].legend()
    ax[1].set_xlabel('Czas(dni)')
    ax[1].set_title('Korelacja r=-0.00000')
    ax[1].set_xlim(0,500)
    ax[1].set_ylim(None,None)


    ax[2].bar([1,2,3,4,5,6,7,8,9], np.abs(pcavec[:,0]), width= 0.3)
    ax[2].set_xticks([1,2,3,4,5,6,7,8,9], header.tolist(), rotation=45)
    ax[2].set_ylabel("Waga")
    ax[2].set_title("Wagi dla składowej numer 0")

    ax[3].bar([1,2, 3, 4, 5, 6, 7, 8, 9], -pcavec[:, 1], width=0.3)
    ax[3].set_xticks([1, 2,3, 4, 5, 6, 7, 8, 9], header.tolist(), rotation=45)
    ax[3].set_ylabel("Waga")
    ax[3].set_title("Wagi dla składowej numer 1")

    plt.show()


def symm():
    def rotate(A, phi):
        Rot = np.array([
            [np.cos(phi), np.sin(phi)],
            [-np.sin(phi), np.cos(phi)]
        ])

        try:
            D = Rot@A
        except ValueError:
            D = Rot@A.T

        return D.T

    A = np.random.randn(1000, 2)
    A[:, 1] = A[:, 1] * 0.05

    B = rotate(A, -np.pi/6)
    C = rotate(A, -np.pi/3)
    data = np.concatenate((B,C), axis=0)
    cov = np.cov(data.T)

    U,S,Vt = np.linalg.svd(cov)
    newB = B @ Vt.T
    newC = C @ Vt.T
    dirs = Vt.T * 2

    plt.figure(figsize = (8,8))
    plt.scatter(B[:,0], B[:,1],marker="$o$", s=20, color = 'blue')
    plt.scatter(C[:,0], C[:,1],marker="$o$", s=20, color = 'blue')
    plt.scatter(newB[:, 0], newB[:, 1], marker="s", s=10, color='red', label = 'Dane po zrzutowaniu na składowe')
    plt.scatter(newC[:, 0], newC[:, 1], marker="s", s=10, color='red')
    plt.plot([0,dirs[0,0]], [0,dirs[1,0]], lw = 4, ls = 'dotted', color = 'red', label = 'Pierwsza składowa')
    plt.plot([0,dirs[0,1]], [0,dirs[1,1]], lw = 4, ls = '--', color = 'red', label = 'Druga składowa')
    plt.grid()
    plt.legend()
    plt.show()

def cats():
    A = np.random.randn(200,2)
    B = np.random.randn(200,2)

    A[:,1] = A[:,0] + np.random.randn(200,)
    B[:,1] = B[:,0] + np.random.randn(200,)

    A = A + np.array([2,-1])
    data = np.concatenate((A,B), axis = 0)
    labels = np.concatenate((np.zeros((200,1)), np.ones((200,1))))
    data = np.concatenate((data, labels), axis = 1) #A
    dataset = pd.DataFrame({'Pierwsza oś danych': data[:,0], 'Druga oś danych': data[:,1], 'kategoria': data[:,2]})

    # g = sns.jointplot(data = dataset,x = "Pierwsza oś danych",y = "Druga oś danych", hue = "kategoria")
    # g.plot_joint(sns.kdeplot, color = ['r','g'], zorder =0, levels = 10)
    # plt.show()
    #
    return data
A = cats()

def LDA(A):
    dataset = pd.DataFrame({'Pierwsza oś danych': A[:, 0], 'Druga oś danych': A[:, 1], 'kategoria': A[:, 2]})

    fst = A[:200,0:2] #kategoria 0
    sec = A[200:,0:2] #kategoria 1

    cov1 = np.cov(fst.T)
    cov2 = np.cov(fst.T)

    C_W = (cov1+cov2)/2 #kowariacja wewnątrzklasowa

    mean1 = np.array(np.mean(fst, axis = 1)).reshape(200,1)
    mean2 = np.array(np.mean(sec, axis = 1)).reshape(200,1)
    means = np.concatenate((mean1, mean2), axis = 1)
    means2 =  np.mean(A[:, 0:2], axis=0)
    C_B = np.cov(means.T) #kowariancja międzyklasowa
    eigval, eigvec = scp.linalg.eigh(C_W, C_B)
    new_data = (A[::,0:2]-means2)@eigvec

    labels = np.array(A[::,-1]).reshape(400,1)
    new_data = np.concatenate((new_data, labels), axis=1)
    dataset2 = pd.DataFrame({'Pierwsza oś po LDA': new_data[:, 0], 'Druga oś po LDA': new_data[:, 1], 'kategoria': new_data[:, 2]})

    #
    # k = sns.jointplot(data = dataset,x = "Pierwsza oś danych",y = "Druga oś danych", hue = "kategoria" )
    # k.plot_joint(sns.kdeplot, color = ['r','g'], zorder =0, levels = 10)
    #
    # g = sns.jointplot(data = dataset2,x = "Pierwsza oś po LDA",y = "Druga oś po LDA", hue = "kategoria")
    # g.plot_joint(sns.kdeplot, color=['r', 'g'], zorder=0, levels=10)

    l = [0  if x<0 else 1 for x in new_data[:,0]]
    l = np.array(l).reshape(400,1)
    results = np.concatenate((l, np.arange(1,401,1).reshape(400,1)), axis = 1)
    acc = [1 if A[i,2] == l[i] else 0 for i in range(400) ]
    perc = sum(acc)/400

    plt.scatter(results[:,1], results[:,0], marker='s', s=20, color= 'blue')
    plt.xticks([0,50,100,150,200,250,300,350,400])
    plt.yticks([0,1],["Klasa 0", "Klasa 1"])
    plt.xlabel("Numer próbki")
    plt.ylabel("Przewidywana klasa")
    plt.title(f"Dokładność: {perc*100}%")
    plt.vlines(x=200, ymin=-0.5, ymax=1.5, color='black', linestyles="dashed")
    plt.show()



def compare(A):
    fst = A[:200, 0:2]  # kategoria 0
    sec = A[200:, 0:2]  # kategoria 1

    cov1 = np.cov(fst.T)
    cov2 = np.cov(fst.T)

    C_W = (cov1 + cov2) / 2  # kowariacja wewnątrzklasowa, w mianowniku

    mean1 = np.array(np.mean(fst, axis=1)).reshape(200, 1)
    mean2 = np.array(np.mean(sec, axis=1)).reshape(200, 1)
    means = np.concatenate((mean1, mean2), axis=1)
    means2 = np.mean(A[:, 0:2], axis=0)
    C_B = np.cov(means.T)  # kowariancja międzyklasowa, w liczniku
    eigval, eigvec = scp.linalg.eigh(C_W, C_B)
    new_data = (A[::, 0:2]-means2) @ eigvec

    labels = np.array(A[::, -1]).reshape(400, 1)
    new_data = np.concatenate((new_data, labels), axis=1)
    dataset2 = pd.DataFrame(
        {'Pierwsza oś po LDA': new_data[:, 0], 'Druga oś po LDA': new_data[:, 1], 'kategoria': new_data[:, 2]})


    return new_data

ldadata = compare(A)


def categorize(new_data, A):
    l = [0 if x < 0 else 1 for x in new_data[:, 0]]
    l = np.array(l).reshape(400, 1)
    results = np.concatenate((l, np.arange(1, 401, 1).reshape(400, 1)), axis=1)
    acc = [1 if A[i, 2] == l[i] else 0 for i in range(400)]
    perc = sum(acc) / 400

    model = LinearDiscriminantAnalysis(solver = 'eigen')
    model.fit(A[::,0:2], A[::, -1])
    pred = model.predict(A[::,0:2]).reshape(400,1)
    goodpred = [0 if x < 0 else 1 for x in pred]


    autores = np.concatenate((pred, np.arange(1, 401, 1).reshape(400, 1)),axis =1)
    autoacc = np.sum(goodpred)/400

    plt.scatter(results[:, 1], results[:, 0], marker='s', s=20, color='blue')
    plt.scatter(autores[:, 1], autores[:, 0], marker='+', s=20, color='red')
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 1], ["Klasa 0", "Klasa 1"])
    plt.xlabel("Numer próbki")
    plt.ylabel("Przewidywana klasa")
    plt.title(f"Dokładność LDA mnualna: {perc * 100}%, Dokładność policzona za pomocą scikit-learn: {autoacc*100}%")
    plt.vlines(x=200, ymin=-0.5, ymax=1.5, color='black', linestyles="dashed")
    plt.show()



def reg(A):

    results = np.zeros((20,2))

    for i in range(20):
        model = LinearDiscriminantAnalysis(solver='eigen', shrinkage= (i+1)/50)
        wholeacc = 0
        for k in range(50):

            indexes = np.arange(0,400,1)
            np.random.shuffle(indexes)
            train = indexes[:350]
            test = indexes[350:]

            train = A[train]
            test = A[test]
            model.fit(train[::, 0:2], train[::, -1])
            pred = model.predict(test[::, 0:2])

            goodpred = [1 if pred[j] == test[j,2] else 0 for j in range(50)]
            acc = np.sum(goodpred)/50
            wholeacc += acc

        wholeacc = wholeacc/50

        results[i,0] = (i+1)/20
        results[i,1] = wholeacc

    print(results)
    plt.figure(figsize=(6,8))
    plt.plot(results[:,0], results[:,1], lw=  2, color = 'red')
    plt.scatter(results[:,0], results[:,1], marker ='s' ,s =20, color = 'red', zorder=20)
    plt.show()

url = 'https://upload.wikimedia.org/wikipedia/ka/1/1c/Stravinsky_picasso.png'
image = io.imread(url)
image = color.rgb2gray(image)
#
# plt.figure(figsize=(8,8))
# plt.imshow(image,cmap='gray')
# plt.show()

def decompose():
    U,s,Vt = np.linalg.svd(image)
    sum = np.zeros_like(image)
    diags = np.zeros((30,2))


    fig, axs = plt.subplots(3,4, figsize = (10,15))
    axs[0,0].imshow(image, cmap = 'gray')
    axs[0,0].set_title('Rozmiar macierzy:' '\n'
                        '(640, 430)' '\n'
                         'rząd: 430')
    for i in range(30):
        diags[i,0] = i
        diags[i,1] = s[i]

    axs[0,1].scatter(diags[:,0], diags[:,1], marker = 's', s = 20, color ='red')
    axs[0,1].plot(diags[:,0], diags[:,1], lw = 1, color ='red')
    axs[0,1].grid()
    axs[0,1].set_title("Wykres osypiska dla rysunku Strawinskiego")
    axs[0,1].set_xlabel("Numer składowej")

    for i in range(4):
        A = np.expand_dims(U[:,i],1)@np.expand_dims(s[i], [0,1])@np.expand_dims(Vt[i,:],0)
        sum += A
        axs[1,i].imshow(A, cmap='gray')
        axs[1,i].set_title(f"L {i}")

        axs[2,i].imshow(sum, cmap='gray')
        axs[2, i].set_title(f"L 0:{i}")

    for e,i in enumerate(axs.flatten()):
        if e not in [0,1]:
            i.set_axis_off()

    plt.tight_layout()
    plt.show()

def reconstruct():
    U, s, Vt = np.linalg.svd(image)
    sum = np.zeros_like(image)

    for i in range(80):
        A = np.expand_dims(U[:, i], 1) @ np.expand_dims(s[i], [0, 1]) @ np.expand_dims(Vt[i, :], 0)
        sum += A

    diff = np.square(image - sum)

    fig, axs = plt.subplots(1,3,figsize = (6,12))
    axs[0].imshow(image, cmap = 'gray')
    axs[0].set_title("Oryginalny obrazek")
    axs[1].imshow(sum, cmap = 'gray')
    axs[1].set_title("Rekonsktrukcja(k=80/430)")
    axs[2].imshow(diff, cmap = 'gray')
    axs[2].set_title("Kwadraty błędów")
    plt.show()

    original = image.nbytes /1024**2
    constr = sum.nbytes /1024**2

    Ubytes = U[:,:80].nbytes/1024**2
    Sbytes = s[:80].nbytes/1024**2
    Vtbyets = Vt[:80,:].nbytes/1024**2

    print(f"Oryginalny obraz: {round(original,2)}MB")
    print(f"Rekonstrukcja: {round(constr,2)}MB")
    print(f"Wektory potrzebne do jego odtworzenia: {round(Ubytes+Sbytes+Vtbyets,2)}MB")
    print(f"Stopień kompresji: {round(100*(Ubytes+Sbytes+Vtbyets)/original,2)}%")


def err():
    U, s, Vt = np.linalg.svd(image)
    sum = np.zeros_like(image)
    plt.figure(figsize = (6,12))
    results = np.zeros((len(s),2))
    for i in range(len(s)):
        A = np.expand_dims(U[:, i], 1) @ np.expand_dims(s[i], [0, 1]) @ np.expand_dims(Vt[i, :], 0)
        sum += A
        error = np.sqrt(np.sum(np.square(image - sum)))
        results[i,0] = i+1
        results[i,1] = error


    gradient = np.diff(results[:,1])

    plt.scatter(results[:,0], results[:,1], marker = 's', s=20)
    plt.plot(results[:,0], results[:,1], lw=2)
    plt.scatter(results[:429,0], gradient, marker='s', s=20, color = 'green')
    plt.plot(results[:429,0], gradient, lw=2,color = 'green')
    plt.xlabel("Rząd rekonstrukcji")
    plt.ylabel("Błąd w stosunku do originał")
    plt.title("Dokładność rekonstrukcji")

    plt.show()

def noise():
    freq = 0.02
    phi = np.pi/3
    noise = np.zeros_like(image)
    x_points = np.linspace(-100, 100, 640)
    y_points = np.linspace(-100,100, 430)

    for iindex,i in enumerate(x_points):
        for jindex,j in enumerate(y_points):
            noise[iindex,jindex] = np.sin(2*np.pi*freq*(i*np.cos(phi) + j*np.sin(phi)))

    noise = (noise - np.min(noise))/(np.max(noise)-np.min(noise))
    noised = image+noise
    noised = (noised - np.min(noised)) / (np.max(noised) - np.min(noised))

    U, s, Vt = np.linalg.svd(noised)
    sum = np.zeros_like(noised)
    diags = np.zeros((30, 2))

    fig, axs = plt.subplots(3, 4, figsize=(10, 15))
    axs[0, 0].imshow(noised, cmap='gray')
    axs[0, 0].set_title('Rozmiar macierzy:' '\n'
                        '(640, 430)' '\n'
                        'rząd: 430')
    for i in range(30):
        diags[i, 0] = i
        diags[i, 1] = s[i]

    axs[0, 1].scatter(diags[:, 0], diags[:, 1], marker='s', s=20, color='red')
    axs[0, 1].plot(diags[:, 0], diags[:, 1], lw=1, color='red')
    axs[0, 1].grid()
    axs[0, 1].set_title("Wykres osypiska dla rysunku Strawinskiego")
    axs[0, 1].set_xlabel("Numer składowej")

    for i in range(4):
        A = np.expand_dims(U[:, i], 1) @ np.expand_dims(s[i], [0, 1]) @ np.expand_dims(Vt[i, :], 0)
        sum += A
        axs[1, i].imshow(A, cmap='gray')
        axs[1, i].set_title(f"L {i}")

        axs[2, i].imshow(sum, cmap='gray')
        axs[2, i].set_title(f"L 0:{i}")

    for e, i in enumerate(axs.flatten()):
        if e not in [0, 1]:
            i.set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.show()


def unnoising():
    freq = 0.02
    phi = np.pi / 3
    noise = np.zeros_like(image)
    x_points = np.linspace(-100, 100, 640)
    y_points = np.linspace(-100, 100, 430)

    for iindex, i in enumerate(x_points):
        for jindex, j in enumerate(y_points):
            noise[iindex, jindex] = np.sin(2 * np.pi * freq * (i * np.cos(phi) + j * np.sin(phi)))

    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    noised = image + noise
    noised = (noised - np.min(noised)) / (np.max(noised) - np.min(noised))

    U, s, Vt = np.linalg.svd(noised)
    sum = np.zeros_like(noised)

    noise_ioslated = np.zeros_like(image)



    for i in range(len(s)):
        A = np.expand_dims(U[:, i], 1) @ np.expand_dims(s[i], [0, 1]) @ np.expand_dims(Vt[i, :], 0)
        if i not in [1,2]:
            sum += A
        else:
            noise_ioslated += A

    #noise_ioslated = (noise_ioslated - np.min(noise_ioslated)) / (np.max(noise_ioslated) - np.min(noise_ioslated))


    cleared = noised-noise_ioslated

    #cleared = (cleared - np.min(cleared)) / (np.max(cleared) - np.min(cleared))

    fig, axs = plt.subplots(1,3,figsize = (15,8))
    axs[0].imshow(noised, cmap='gray')
    axs[0].set_title('Zaszumiony obrazek')

    axs[1].imshow(noise_ioslated, cmap='gray')
    axs[1].set_title('Szum(składowe 1 i 2)')

    axs[2].imshow(cleared, cmap='gray')
    axs[2].set_title('Obrazek po usunięciu szumu')

    plt.show()

unnoising()