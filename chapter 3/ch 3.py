import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import plotly.graph_objs as go



def linear_combination(x,w):
    r = np.shape(x)[0]
    c = np.shape(x)[1]
    final = np.zeros((np.shape(w)[0],np.shape(w)[1]))

    for i in range(r):
        for j in range(c):
            p = x[j][i] * w[0][j]
            p_c = copy.deepcopy(p)
            final[0][i] += p_c

    return final

def plot_points_2D(points):
    plt.scatter([x[0] for x in points], [x[1] for x in points])
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.grid(True)
    plt.show()


def plot_points_3D(points):
    fig = go.Figure(data = [go.Scatter3d(x = [x[0] for x in points],
                                         y = [x[1] for x in points],
                                         z = [x[2] for x in points],
                                         mode = 'markers')])
    fig.show()



x = np.array([[1.5,2.5,0.5],[0,2,2]])

w1 = [random.uniform(-4,4) for i in range(100)]
w2 = [random.uniform(-4,4) for i in range(100)]
points1 = [x[0] * weight for weight in w1]
points2 = [x[1] * weight for weight in w2]


points = [x+y for (x,y) in zip(points1, points2)]

print(points)

#print(points, len(points))
#plot_points_2D(points)
plot_points_3D(points)