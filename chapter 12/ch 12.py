import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
df = pd.read_csv(url,sep =',', encoding='unicode_escape')

#1
def drop():
    df1 = df.drop(df[df['Rainfall(mm)'] == 0].index) #dropuje elementy df gdzie rainfall == 0
    #można też df1 = df.loc[df['Rainfall(mm)'] != 0]

    desmat = df1[['Rainfall(mm)', 'Seasons']]
    desmat.replace(['Autumn', 'Winter', 'Summer', 'Spring'], [0, 0, 1, 1, ], inplace=True)
    desmat = desmat.to_numpy()


    desmat = np.hstack((desmat, np.ones((desmat[:,1].shape[0],1))))
    y = df1[['Rented Bike Count']].to_numpy()


    beta = np.linalg.lstsq(desmat,y,rcond=None)#least square
    real = df1['Rented Bike Count'].to_list()
    pred = desmat@beta[0]

    SST = np.sum((y - np.mean(y)) ** 2)
    SSE = np.sum((y - pred) ** 2)

    RS = 1 -(SSE/SST)

    plt.figure(figsize=(6,8))
    plt.scatter(real, pred)
    plt.xlabel('Rzeczywista liczba rowerów')
    plt.ylabel('Przewidywana liczba rowerów')
    plt.title(f'Dopasowanie modelu (R^2): {RS}')
    plt.show()

#2
def predict():

    desmat = df[['Rainfall(mm)', 'Temperature(°C)']].to_numpy()
    desmat = np.hstack((desmat, np.ones((desmat.shape[0],1))))

    y = df['Rented Bike Count'].to_numpy()
    beta = np.linalg.lstsq(desmat, y, rcond=None)

    pred = desmat@beta[0]

    SST = np.sum(np.square(y - np.mean(y)))
    SSE = np.sum(np.square(y - pred))

    RSQ = 1 - (SSE/SST)

    plt.figure(figsize=(6, 15))
    plt.scatter(y, pred)
    plt.xlabel('Rzeczywista liczba rowerów')
    plt.ylabel('Przewidywana liczba rowerów')
    plt.title(f'Dopasowanie modelu (R^2): {RSQ}')
    plt.show()

def create_matrix():
    desmat = df[['Rainfall(mm)', 'Temperature(°C)']].to_numpy()
    desmat = np.hstack((desmat, np.ones((desmat.shape[0], 1))))
    D1 = desmat
    lincomb = (random.randint(1,4)*df['Rainfall(mm)'].to_numpy()) +\
              (random.randint(1,4)*df['Temperature(°C)'].to_numpy())
    lincomb = lincomb.reshape(lincomb.shape[0],1)
    desmat = np.hstack((desmat, lincomb))
    y = df['Rented Bike Count'].to_numpy()

    B_one =  np.linalg.inv(desmat.T@desmat)@desmat.T@y
    YH1 = desmat@B_one
    R1 = np.corrcoef(y.T, YH1.T)[0,1]**2

    B_two = np.linalg.lstsq(desmat,y)[0]
    YH2 = desmat@B_two
    R2 = np.corrcoef(y.T, YH2.T)[0, 1] ** 2

    model = sm.OLS(y,desmat).fit()
    B_three = model.params
    R3 = model.rsquared

    # print('DOPASOWANIE MODELU DO DANYCH:')
    # print(f'odwrotność lewostronna:{R1}')
    # print(f'np.lstsq:{R2}')
    # print(f'statsmodel:{R3}')
    #
    # print('WSPÓCZYNNIKI BETA:')
    # print(f'odwrotność lewostronna:{B_one}')
    # print(f'np.lstsq:{B_two}')
    # print(f'statsmodel:{B_three}')

    return desmat, D1



def regularize(A,B):

    regularizerA = np.sum(A.T@A)* np.eye(4)
    regularizerB = np.sum(B.T@B)* np.eye(3)
    print(regularizerA, regularizerB)
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

A,B = create_matrix()
regularize(A,B)

def polynomial():
    year = [1534, 1737, 1803, 1928, 1960, 1975, 1987, 2023, 2057, 2100]
    doubleTime = [697, 594, 260, 125, 76, 47, 37, 48, 70, 109]

    _, ax = plt.subplots(2,5, figsize =(14,5))
    axes = ax.flatten()

    for pow in range(10):
        beta = np.polyfit(year,doubleTime,pow)
        yhat = np.polyval(beta, year)

        axes[pow].scatter(year, doubleTime, marker ='s', s=40, color = 'black')
        axes[pow].plot(year, doubleTime, lw = 1, color = 'black')

        axes[pow].scatter(year, yhat, marker='o', s=12, color='gray')
        axes[pow].plot(year, yhat,'--', lw=1, color='gray')
        axes[pow].set_title(f"Stopień = {pow}")


    plt.tight_layout()
    plt.show()

def search():
    numcourses = [13, 4, 12, 3, 14, 13, 12, 9, 11, 7, 13, 11, 9, 2, 5, 7, 10, 0, 9, 7]
    happiness = [70, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 70]
    X = np.hstack((np.ones((20,1)), np.array(numcourses, ndmin = 2).T))

    beta = np.linalg.lstsq(X,happiness, rcond=None)[0]
    yHat = X@beta
    print(beta)

    intercepts = np.linspace(0,80,100)
    slopes = np.linspace(0,6,100)

    SSEs = np.zeros((len(intercepts),len(slopes)))

    for i, inter in enumerate(intercepts): #iterujemy po kombinacjach nachylenia i pkt, przecięcia
        for j, slp in enumerate(slopes):

            yHat = X@np.array([inter, slp]).T #oblicznamy przewidywania dla danych współczynników

            SSEs[i,j] = np.sum(np.square(yHat - happiness)) #obliczamy SSE dla każdej z tej pary

    #zwracaja indes na którym byłoby indexy zwracane przez SSE(zfatowana tablica)
    # gdyby znajdowały się w wabtlcy o wyamirach SSEs.shape, używamy tego bo aragmin automatycznie flattuje
    #tablice i zwara index w "liście" wiec musimy go odsukać w tablicy o rozmairz SSEs.shape
    i, j = np.unravel_index(np.argmin(SSEs), SSEs.shape)
    # np.argmin zwraca index najmniejszej wartości
    empIntercept, empSlope = intercepts[i], slopes[j]

    plt.figure(figsize=(6, 6))
    plt.imshow(SSEs, vmin=2000, vmax=3000, #ustawiając vmin i vmax możemyu ustawić kolory i gdize zostanie narysowana heamata
               extent=(slopes[0], slopes[-1], intercepts[0], intercepts[-1]),#mapuje osie na warotści sloe i intercept
               origin='lower', aspect='auto', cmap='gray')
    plt.plot(empSlope, empIntercept, 'o', color=[1, .4, .4], markersize=12, label='Minimum z przeszukiwania\nsiatki')
    plt.plot(beta[1], beta[0], 'x', color=[.4, .7, 1], markeredgewidth=4, markersize=10,
             label='Rozwiązanie\nanalityczne')
    plt.colorbar()
    plt.xlabel('Współczynnik nachylenia')
    plt.ylabel('Punkt przecięcia')
    plt.title('SSE (dopasowanie modelu do danych)')
    plt.legend()
    plt.savefig('rys12.8.png', dpi=300)
    plt.show()


#7
def gowno():
    numcourses = [13, 4, 12, 3, 14, 13, 12, 9, 11, 7, 13, 11, 9, 2, 5, 7, 10, 0, 9, 7]
    happiness = [70, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 70]
    X = np.hstack((np.ones((20, 1)), np.array(numcourses, ndmin=2).T))

    beta = np.linalg.lstsq(X, happiness, rcond=None)[0]
    yHat = X @ beta
    print(beta)

    intercepts = np.linspace(0, 80, 100)
    slopes = np.linspace(0, 6, 100)

    Rs = np.zeros((len(intercepts), len(slopes)))

    for i, inter in enumerate(intercepts):  # iterujemy po kombinacjach nachylenia i pkt, przecięcia
        for j, slp in enumerate(slopes):
            yHat = X @ np.array([inter, slp]).T  # oblicznamy przewidywania dla danych współczynników
            Rs[i, j] = np.corrcoef(np.array(happiness),yHat)[0,1]**2  # obliczamy SSE dla każdej z tej pary

    # zwracaja indes na którym byłoby indexy zwracane przez SSE(zfatowana tablica)
    # gdyby znajdowały się w wabtlcy o wyamirach SSEs.shape, używamy tego bo aragmin automatycznie flattuje
    # tablice i zwara index w "liście" wiec musimy go odsukać w tablicy o rozmairz SSEs.shape
    i, j = np.unravel_index(np.argmin(Rs), Rs.shape)
    # np.argmin zwraca index najmniejszej wartości
    empIntercept, empSlope = intercepts[i], slopes[j]

    plt.figure(figsize=(6, 6))
    plt.imshow(Rs, vmin=2000, vmax=3000,
               # ustawiając vmin i vmax możemyu ustawić kolory i gdize zostanie narysowana heamata
               extent=(slopes[0], slopes[-1], intercepts[0], intercepts[-1]),
               # mapuje osie na warotści sloe i intercept
               origin='lower', aspect='auto', cmap='gray')
    plt.plot(empSlope, empIntercept, 'o', color=[1, .4, .4], markersize=12, label='Minimum z przeszukiwania\nsiatki')
    plt.plot(beta[1], beta[0], 'x', color=[.4, .7, 1], markeredgewidth=4, markersize=10,
             label='Rozwiązanie\nanalityczne')
    plt.colorbar()
    plt.xlabel('Współczynnik nachylenia')
    plt.ylabel('Punkt przecięcia')
    plt.title('Rs (dopasowanie modelu do danych)')
    plt.legend()
    plt.savefig('rys12.9.png', dpi=300)
    plt.show()

