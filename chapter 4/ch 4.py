import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
import timeit
import time
import threading

def percos(x,y, center = False):
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    dem = (np.linalg.norm(x_c) * np.linalg.norm(y_c))
    per = np.dot(x_c, y_c.T) / dem
    return per

def times():
    x = [random.uniform(-9999999,9999999) for i in range(5000)]
    y = [random.uniform(-9999999,9999999) for i in range(5000)]
    start = time.time()
    for i in range(1000):
        percos(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
    end = time.time()
    print(f"own: {(end - start)}")
    for i in range(1000):
        np.corrcoef(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
    end2 = time.time()
    print(f"numpy: {end2 - end}")


def convert(kernel, series):
    filtered = []
    r = np.shape(series)[0] - np.shape(kernel)[0] + 1

    for i in range(r):
        filtered.append(np.dot(kernel,np.array([series[j].item() for j in range(i, i + len(kernel))]).T
                               ))

    plt.subplot(1,3,1)
    plt.scatter(range(np.shape(kernel)[0]),kernel)
    plt.plot(range(np.shape(kernel)[0]),kernel, linewidth = 1)
    plt.xlim(-15,15)
    plt.ylim(-3,3)
    plt.title('Kernel')


    plt.subplot(1,3,2)
    plt.scatter(range(np.shape(series)[0]), series)
    plt.plot(range(np.shape(series)[0]), series, linewidth=1)
    plt.xlim(-3, 300)
    plt.ylim(-3, 3)
    plt.title('Signal')

    plt.subplot(1,3,3)
    plt.scatter(range(np.shape(filtered)[0]), filtered)
    plt.plot(range(np.shape(filtered)[0]), filtered, linewidth=1)
    plt.xlim(-3, 300)
    plt.ylim(-3, 3)
    plt.title('Filtered signal')

    plt.show()


    return filtered

kernel = np.array([-1,-1])
series = np.random.normal(loc = 0.0, scale= 1.0, size = 300)

def k_means(k, data,n=5):
    choices = np.random.choice(len(data), k, replace = False)
    centroids = data[choices,:]
    distances = np.zeros((np.shape(data)[0], k))

    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color = []
    for _ in range(k):
        c = random.sample(colors,1)[0]
        color.append(c)
        colors.remove(c)

    for e in range(n):


        #distances from each point to each centroid( column 1 is all points distances from centroid 1 etc)
        for i in range(k):
            distances[:,i] = np.sum(((data - centroids[i])**2), axis = 1)

        #determine which centroid is the point closest to 1 - centroid 1, 2 - centroid 2 etc
        group_ids = np.argmin(distances, axis=1)

        new_centroids = np.zeros((np.shape(centroids)[0], np.shape(centroids)[1]))
        #create groups for storing the centroids coordinates for ploting
        groups = [[] for _ in range(k)]
        #calculate sum off all points closest to centroid
        for i, element in enumerate(data):
            new_centroids[group_ids[i]] += element
            groups[group_ids[i]].append(element)
        #update the centroids coordinates
        #print(groups)
        for i in range(k):
            new_centroids[i] = new_centroids[i]/ group_ids.tolist().count(i)
        centroids = new_centroids
        #print(groups[1][0][0])
        #draw scatterplot for points, each category is in different colors, centroids are red
        for m in range(k):
            plt.scatter([groups[m][j][0] for j in range(len(groups[m]))],
                        [groups[m][j][1] for j in range(len(groups[m]))], color=color[m])
            plt.plot([groups[m][j][0] for j in range(len(groups[m]))],
                        [groups[m][j][1] for j in range(len(groups[m]))], color=color[m], linewidth  = 1)
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)

        plt.show()


    return centroids
data = np.random.uniform(low = -100,high = 100, size = (150,2))
k_means(9,data)

