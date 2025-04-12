import matplotlib.animation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import convolve2d
from skimage import io, color
import os

rc('animation', html = 'jshtml')


#1,2
def npcov():
     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
     df = pd.read_csv(url, sep = ',', header = None)

     df.columns = [ 'state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize', 'racepctblack', 'racePctWhite',
     'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage',
     'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap',
     'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy',
     'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par',
     'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
     'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup',
     'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc',
     'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian',
     'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85',
     'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop',
     'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea',
     'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'ViolentCrimesPerPop',
      ]

     data = df._get_numeric_data()
     print(data.shape)
     dataMat = data.drop(['state','fold'],axis=1).values

     mean = dataMat.mean(axis = 0)
     dataMat = dataMat - mean

     cov = dataMat.T @ dataMat
     print(cov.shape) #(100,100)
     cov /= (cov.shape[0] - 1)

     standev= np.diag(1/(dataMat.T @ dataMat).std(axis = 0))
     corr = standev @ cov @ standev

     plt.subplot(1,3,1)
     plt.imshow(cov,cmap='gray')
     plt.xlim(0,100)
     plt.ylim(100,0)
     plt.colorbar()
     plt.title('Macierz korelacji danych')

     plt.subplot(1,3,2)
     plt.imshow(np.cov(dataMat, rowvar = False),cmap='gray')
     plt.xlim(0,100)
     plt.ylim(100,0)
     plt.colorbar()
     plt.title('Macierz korelacji danych numpy')

     plt.subplot(1,3,3)
     plt.imshow(np.cov(dataMat, rowvar = False) - cov,cmap='gray')
     plt.xlim(0,100)
     plt.ylim(100,0)
     plt.colorbar()
     plt.title('rożnica macierzy')


     plt.show()

def transform_line():
     th = np.pi / 2.5
     #macierz transformacji
     T = np.array([
                    [ np.cos(th), np.sin(th)],
                    [-np.sin(th), np.cos(th)]
     ])

     x = np.linspace(-1,1,20)

     origin = np.vstack((np.zeros(x.shape),x)) # vertical line(points)

     transformed = T @ origin

     plt.figure(figsize = (6,6))
     plt.plot(origin[0,:], origin[1,:], color = 'red', label = 'Przed obrotem', linewidth = '2')
     plt.plot(transformed[0,:], transformed[1,:], color = 'blue', label = 'Po obrocie', linewidth = '2')
     plt.axis('square')
     plt.xlim([-1.2,1.2])
     plt.ylim([-1.2,1.2])
     plt.legend()
     plt.title(f'Obrót o {int(np.rad2deg(th))} stopni')
     plt.show()

def wobbly_circle():
     def update(th):
          T = np.array([
               [1, 1-th],
               [1, 0]
          ])
          P = T @ points

          #aktualizacja poszczególnych kropek uchywtu, na podstawie przetransformowanych punktów( punkty są wysyłane do funkcji sin(x), cos(x) uochywtu
          plth.set_xdata(P[0,:])
          plth.set_ydata(P[1,:])
          # ax.relim()
          # ax.autoscale_view()

          return plth

     theta = np.linspace(0, 2*np.pi, 100)# 100 punktów w zakresie od 0 do 2pi
     points = np.vstack((np.cos(theta), np.sin(theta))) # x = cos(theta), y = sin(theta), wektor zawierający 100 par x i y

     fig, ax = plt.subplots(1, figsize = (12,6)) #tworzymy 1 wykres o rozmiarze 6,6
     #fig reprezentuje pojemnik na wsyzskite wykresy
     #ax reprezentuje faktyczny wykres na którym będziemy rysować, w przypdaku większej ilości wykresów
     #ax będize tablicą

     plth, = ax.plot(points[0,:], points[1,:], 'ko')

     phi = np.linspace(-1, 1-1/40, 40) **2
     ax.set_aspect('equal')
     ax.set_xlim([-2, 2])
     ax.set_ylim([-2, 2])


     anim = matplotlib.animation.FuncAnimation(fig, update, phi, interval = 20, repeat = True)
     plt.show()

#3
def circle():

     theta = np.linspace(0, 2 * np.pi, 18)
     points = np.vstack((np.sin(theta), np.cos(theta)))

     T = np.array([
          [1, 0.5],
          [0, 0.5]
     ])

     plt.figure(figsize=(6,6))
     plt.plot(points[0,:], points[1,:], 'ko', color= 'red', label = 'przed przesunięciem')



     points = T @ points

     plt.plot(points[0,:], points[1,:], 's', color = 'green', label = 'po przesunięciu')
     plt.axis('square')
     plt.xlim([-2, 2])
     plt.ylim([-2, 2])
     plt.legend()
     plt.show()

#4
def DNA():
     def update(phi):
          T = np.array([
               [1- phi/3, 0],
               [0,       phi]
          ])

          P = T @ points

          hold.set_xdata(P[0,:])
          hold.set_ydata(P[1,:])

          return hold

     #punkty które rysujemy na początku
     theta = np.linspace(0, 2 * np.pi,100)
     points = np.vstack((theta,np.sin(theta)))

     fig, ax = plt.subplots(1, figsize = (6,10))

     # interwały w krórych będzie zmieniał się ka
     phi = np.linspace(-1,1, 100)** 2
     hold, = ax.plot(points[0, :], points[1, :], 'ko', color = 'red')


     ax.set_aspect('equal')

     anim = matplotlib.animation.FuncAnimation(fig, update, phi, interval= 1, repeat = True)
     plt.show()

def manual_smoothing():
     imgN = 20
     #macierz liczb losowych( z rozkałdau noramlnego) 20x20
     image = np.random.randn(imgN, imgN)

     kernelN = 7
     Y,X = np.meshgrid(np.linspace(-3,3, kernelN), np.linspace(-3,3, kernelN))


     kernel = np.exp( -(X**2+Y**2)/7 )
     kernel /= np.sum(kernel)

     halfKr = kernelN//2


     #placeholder for image after convolution(same shape as imagePad with additonal zeros, so the kenrel can go thorugh whole
     # image without the index error)
     convoutput = np.zeros((imgN + kernelN - 1,imgN + kernelN - 1))



     imagePad = np.zeros(convoutput.shape)
     #fill the image pad(26,26 matrix filled with zeros) with random numbers( only the 3-23 rows and columns), rest stay zeros
     imagePad[halfKr:-halfKr,halfKr:-halfKr] = image

     print(imagePad)


     for row in range(halfKr, imgN + halfKr): #range from 3 to 23 (23 excluded)
          for col in range(halfKr, imgN + halfKr): # range from 3 to 23(skipping first and last 3 elements)
               ImagePiece = imagePad[row - halfKr: row + halfKr + 1, #i -3, i+4: 0-7,1-8,2-9
                                     col - halfKr: col + halfKr + 1] #i -3, i+4: 0-7,1-8,2-9, getting 7x7 piece each time cuz kernel is 7x7

               product = np.sum(ImagePiece * kernel)
               convoutput[row, col] = product

     #getting rid of zeros
     convoutput = convoutput[halfKr:-halfKr,halfKr:-halfKr]

     #convolution by Scipy
     convoutput2 = convolve2d(image, kernel, mode = 'same')

     fig,ax = plt.subplots(2,2, figsize = (8,8))

     ax[0,0].imshow(image)
     ax[0,0].set_title("Obrazek")

     ax[0,1].imshow(kernel)
     ax[0,1].set_title("Jądro")

     ax[1,0].imshow(convoutput)
     ax[1,0].set_title('Obrazek wygładzony "ręcznie" ')

     ax[1,1].imshow(convoutput2)
     ax[1,1].set_title('Obrazek wygładzony z wykorzystaniem Scipy')
     for i in ax.flatten(): i.axis('off')
     plt.savefig('rys7.4b.png', dpi=300)

     bathtub = io.imread('https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')
     fig = plt.figure(figsize = (10,6))


     bathtub2d = color.rgb2gray(bathtub)
     bathtub.shape
     plt.imshow(bathtub2d)

     kernelN = 29
     Y, X = np.meshgrid(np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN))
     kernel = np.exp(-(X ** 2 + Y ** 2) / 20)
     kernel = kernel / np.sum(kernel)  # normalizacja

     smooth_bathtub = convolve2d(bathtub2d, kernel, mode='same')

     fig = plt.figure(figsize=(10, 6))
     plt.imshow(smooth_bathtub)

     plt.show()
#5
def smoothen():
     bathtub = io.imread(
          'https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

     kernelN = 29
     Y, X = np.meshgrid(np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN))
     kernel = np.exp(-(X ** 2 + Y ** 2) / 29)
     kernel = kernel / np.sum(kernel)  # normalizacja

     smoothened_image = np.zeros(bathtub.shape)

     fig, ax = plt.subplots(1,2,figsize = (6,8))

     for i in range(3): smoothened_image[:,:,i] = convolve2d(bathtub[:,:,i], kernel, mode = 'same').astype(np.uint8)

     smoothened_image = smoothened_image.astype(np.uint8)

     ax[0].imshow(bathtub)
     ax[0].set_title("Przed wygładaniem")

     ax[1].imshow(smoothened_image)
     ax[1].set_title("Po wygładzaniem")

     plt.show()
#6
def smoothen2():
     bathtub = io.imread(
          'https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

     fig, ax = plt.subplots(2, 3, figsize=(8, 10))

     kernelN = 29
     smoothened_image = np.zeros(bathtub.shape)
     kernrelwidths = [0.5,5,50]
     channels = ['R','G','B']
     for i in range(3):
          Y, X = np.meshgrid(np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN))
          kernel = np.exp(-(X ** 2 + Y ** 2) / kernrelwidths[i])
          kernel = kernel / np.sum(kernel)  # normalizacja
          smoothened_image[:, :, i] = convolve2d(bathtub[:, :, i], kernel, mode='same').astype(np.uint8)
          ax[1, i].imshow(kernel)
          ax[1, i].set_title(f"kernel {i} (Kanał {channels[i]})")



     smoothened_image = smoothened_image.astype(np.uint8)

     ax[0,0].imshow(bathtub)
     ax[0,0].set_title("Przed wygładaniem")

     ax[0,1].imshow(smoothened_image)
     ax[0,1].set_title("Po wygładzaniem")

     plt.show()

def smoothen3():
     bathtub = io.imread(
          'https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

     bathtub2d = color.rgb2gray(bathtub)
     vertical = np.array([
          [1,0,-1],
          [1,0,-1],
          [1,0,-1]
     ])

     horizontal = np.array([
          [ 1 , 1 ,1],
          [ 0 , 0 ,0],
          [-1, -1,-1]
     ])

     fig, ax = plt.subplots(2, 2, figsize=(6, 8))

     smoothened_image1 = convolve2d(bathtub2d, vertical, mode = 'same')
     smoothened_image2 = convolve2d(bathtub2d, horizontal, mode = 'same')

     print(smoothened_image2.shape)

     ax[0, 0].imshow(vertical, cmap = 'gray')
     ax[0, 0].set_title('Jądro pionowe')
     ax[0, 0].set_yticks(range(3))

     ax[0,1].imshow(horizontal, cmap = 'gray')
     ax[0,1].set_title('Jądro poziome')
     ax[0,1].set_yticks(range(3))

     ax[1,0].imshow(smoothened_image1, cmap = 'gray',vmin=0,vmax=.01)
     ax[1,0].set_title("Wygładzenie jądrem pionowym")

     ax[1,1].imshow(smoothened_image2, cmap = 'gray',vmin=0,vmax=.01)
     ax[1,1].set_title("Wygładzenie jądrem poziomym")

     plt.show()

smoothen3()
